"""Shared-backbone writing, querying, composition and soft-token-conditioned loss."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import ByteTokenizer, HFBackbone, RecurrentBackbone, TinyBackbone
from .compaction import SyntheticCompactor, contribution_loss, mean_and_mass, storage_noise
from .config import Config
from .readers import MultiSpaceReader, SetReader
from .routing import cosine_scores, group_plan_loss


@dataclass
class ForwardResult:
    loss: Tensor
    nll: Tensor
    routing_loss: Tensor
    compaction_loss: Tensor
    selected: list[list[int]]


class MemoryAgent(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        m, r = config.model, config.memory
        if m.backend == "tiny":
            base = TinyBackbone(m.tiny_width, m.tiny_layers, m.tiny_heads)
            base.gradient_checkpointing = m.gradient_checkpointing
            self.tokenizer = ByteTokenizer()
            self.resolved_revision = "local-tiny"
        else:
            base = HFBackbone(m.model_id, revision=m.revision,
                              gradient_checkpointing=m.gradient_checkpointing)
            self.tokenizer = base.tokenizer
            self.resolved_revision = base.resolved_revision
        self.backbone = RecurrentBackbone(base, m.loops)
        self.width = base.width
        if m.freeze_backbone:
            for p in self.backbone.base.parameters():
                p.requires_grad_(False)
        self.write_slots = nn.Parameter(torch.randn(r.write_slots + 1, self.width) * 0.02)
        self.key_head = nn.Linear(self.width, r.key_dim, bias=False)
        self.value_head = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, self.width))
        self.query_head = nn.Linear(self.width, r.key_dim, bias=False)
        self.address_maps = nn.ModuleList([nn.Linear(r.key_dim, r.key_dim, bias=False) for _ in r.payload_dims])
        self.query_maps = nn.ModuleList([nn.Linear(r.key_dim, r.key_dim, bias=False) for _ in r.payload_dims])
        canonical_dim = r.write_slots * self.width
        self.codecs = nn.ModuleList([
            nn.Identity() if d == canonical_dim else nn.Linear(canonical_dim, d)
            for d in r.payload_dims
        ])
        options = dict(width=r.reader_width, slots=r.read_slots, rounds=r.reader_rounds,
                       kind=r.reader, chunk_size=r.chunk_size, checkpoint_chunks=r.checkpoint_chunks)
        self.reader = (SetReader(r.payload_dims[0], r.key_dim, self.width, **options)
                       if len(r.payload_dims) == 1 else
                       MultiSpaceReader(r.payload_dims, r.key_dim, self.width, **options))
        if config.train.arm == "direct_latent" and r.payload_dims != [canonical_dim]:
            raise ValueError("direct_latent requires a single identity-width stored payload")
        self.compactor = (SyntheticCompactor(r.payload_dims[0], r.reader_width, r.compact_records)
                          if r.compaction == "synthetic" else None)
        self.memory_norm = nn.LayerNorm(self.width)
        self.memory_gate = nn.Parameter(torch.tensor(-1.0))

    @property
    def device(self) -> torch.device:
        return self.write_slots.device

    def text_ids(self, text: str, *, source: bool = False) -> Tensor:
        limit = self.config.train.max_source_tokens if source else self.config.train.max_prompt_tokens
        ids = self.tokenizer.encode(text, add_special_tokens=source)
        if not ids:
            ids = [self.tokenizer.bos_token_id or 1]
        # Fail explicitly rather than silently deleting a decisive rule/identifier.
        if len(ids) > limit:
            raise ValueError(f"Text uses {len(ids)} tokens, exceeding configured limit {limit}")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def prompt_ids(self, query: str, support_text: str = "") -> Tensor:
        content = ("Prior experiences:\n" + support_text + "\n\n" if support_text else "") + query
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        return self.text_ids(text)

    def produce(self, source_ids: Tensor) -> tuple[Tensor, ...]:
        """One key and fixed canonical slots, then per-space keys/stored payloads.

        Returns (key_s0, payload_s0, key_s1, payload_s1, ...). Capturing this
        boundary replays the writer AND storage transforms, not just raw values.
        """
        tokens = self.backbone.embed(source_ids)
        embeddings = torch.cat((tokens, self.write_slots[None].expand(tokens.shape[0], -1, -1)), 1)
        hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long))
        tail = hidden[:, -(self.config.memory.write_slots + 1):]
        key = F.normalize(self.key_head(tail[:, 0]), dim=-1)
        canonical = self.value_head(tail[:, 1:]).flatten(1)
        output = []
        for address, codec in zip(self.address_maps, self.codecs, strict=True):
            output.extend((F.normalize(address(key), dim=-1), codec(canonical)))
        return tuple(output)

    def query(self, prompt_ids: Tensor) -> Tensor:
        embeddings = self.backbone.embed(prompt_ids)
        # Query uses ONLY the prompt, never teacher-forced target tokens.
        h = self.backbone.hidden(embeddings, torch.ones_like(prompt_ids), loops=1)
        return F.normalize(self.query_head(h[:, -1]), dim=-1)

    def read_tokens(self, payloads: list[Tensor], query: Tensor, *, compact: bool = False,
                    ablate_values: bool = False) -> tuple[Tensor, Tensor]:
        r = self.config.memory
        values = [v[None] for v in payloads]
        if ablate_values:
            values = [torch.zeros_like(v) for v in values]
        if self.training:
            values = [storage_noise(v, noise_std=r.noise_std, quantization_step=r.quantization_step) for v in values]
        weights = [v.new_ones(v.shape[:2]) for v in values]
        auxiliary = query.sum() * 0
        if isinstance(self.reader, MultiSpaceReader):
            return self.reader(values, query, weights), auxiliary
        if compact and values[0].shape[1] > 0:
            if r.compaction == "mean":
                compact_values, compact_weights = mean_and_mass(values[0], weights[0])
            elif r.compaction == "synthetic":
                compact_values, compact_weights = self.compactor(values[0], weights[0])
            else:
                raise ValueError("Compacted read requested without a compactor")
            auxiliary = contribution_loss(self.reader, values[0], weights[0],
                                          compact_values, compact_weights, query)
            return self.reader(compact_values, query, compact_weights).tokens, auxiliary
        return self.reader(values[0], query, weights[0]).tokens, auxiliary

    def _context(self, prompt: Tensor, memory: Tensor | None) -> Tensor:
        context = self.backbone.embed(prompt)
        if memory is not None and memory.shape[1] > 0:
            scale = context.detach().float().square().mean().sqrt().clamp_min(1e-3)
            tokens = self.memory_norm(memory) * scale.to(memory.dtype) * self.memory_gate.sigmoid()
            context = torch.cat((context, tokens.to(context.dtype)), 1)
        return context

    def conditioned_nll(self, prompt: Tensor, target: Tensor, memory: Tensor | None) -> Tensor:
        context = self._context(prompt, memory)
        # First target is predicted from last context token; no label shift ambiguity.
        embeddings = torch.cat((context, self.backbone.embed(target[:, :-1])), 1)
        hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long))
        predictive = hidden[:, context.shape[1] - 1:]
        logits = self.backbone.logits(predictive).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    def target_ids(self, answer: str) -> Tensor:
        ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        return torch.tensor([ids], device=self.device)

    def forward(self, prompt: Tensor, target: Tensor, records: list[tuple[Tensor, ...]],
                required: list[int], *, step: int = 0, arm: str | None = None,
                compact: bool = False, ablate_values: bool = False) -> ForwardResult:
        t, r = self.config.train, self.config.memory
        arm = arm or t.arm
        zero = self.memory_gate * 0
        if arm in {"no_memory", "oracle_text"}:
            nll = self.conditioned_nll(prompt, target, None)
            return ForwardResult(nll, nll, zero, zero, [])
        q = self.query(prompt)
        selected, payloads = [], []
        routing = zero
        for space, dim in enumerate(r.payload_dims):
            keys = torch.cat([record[2 * space] for record in records], 0) if records else q.new_empty(0, r.key_dim)
            values = torch.cat([record[2 * space + 1] for record in records], 0) if records else q.new_empty(0, dim)
            scores = cosine_scores(F.normalize(self.query_maps[space](q), dim=-1), keys)[0]
            if t.retrieval == "learned" and required and records:
                routing = routing + group_plan_loss(scores, [tuple(required)]) / len(r.payload_dims)
            oracle = t.retrieval == "oracle" or (self.training and step < t.routing_warmup)
            indices = required if oracle else torch.argsort(scores, descending=True, stable=True)[:r.neighbors[space]].tolist()
            selected.append(list(indices))
            payloads.append(values[indices])
        present = any(p.shape[0] for p in payloads)
        if arm == "direct_latent":
            memory = payloads[0].reshape(1, -1, self.width) if present else None
            auxiliary = zero
        elif present:
            memory, auxiliary = self.read_tokens(payloads, q, compact=compact, ablate_values=ablate_values)
        else:
            memory, auxiliary = None, zero
        nll = self.conditioned_nll(prompt, target, memory)
        loss = nll + t.routing_weight * routing + r.compaction_loss_weight * auxiliary
        return ForwardResult(loss, nll, routing, auxiliary, selected)

    @torch.no_grad()
    def generate_with_payloads(self, prompt: Tensor, payloads: list[Tensor] | None,
                               max_new_tokens: int = 24) -> str:
        """Reference greedy decoding: recomputes prefixes; no hybrid cache mutation."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        memory = None
        if payloads is not None and any(p.shape[0] for p in payloads):
            q = self.query(prompt)
            memory, _ = self.read_tokens(payloads, q)
        context = self._context(prompt, memory)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            embeddings = context
            if generated:
                ids = torch.tensor([generated], device=self.device)
                embeddings = torch.cat((context, self.backbone.embed(ids)), 1)
            hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long))
            next_id = int(self.backbone.logits(hidden[:, -1]).argmax(-1).item())
            if next_id == self.tokenizer.eos_token_id:
                break
            generated.append(next_id)
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
