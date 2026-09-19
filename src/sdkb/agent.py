"""Shared-backbone writing, querying, composition and soft-token-conditioned loss."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .backbones import ByteTokenizer, HFBackbone, RecurrentBackbone, TinyBackbone
from .compaction import SyntheticCompactor, contribution_loss, storage_noise, compact_view
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
    raw_nll: Tensor | None = None
    compact_nll: Tensor | None = None
    behavior_kl: Tensor | None = None
    read_count: int = 1


class SDKBAgent(nn.Module):
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
        from .text import render_prompt
        text = render_prompt(self.tokenizer, query, support_text)
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

    def query(self, prompt_ids: Tensor, memory: Tensor | None = None) -> Tensor:
        embeddings = self._context(prompt_ids, memory)
        # Query uses ONLY the prompt, never teacher-forced target tokens.
        h = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long), loops=1)
        return F.normalize(self.query_head(h[:, -1]), dim=-1)

    def _prepare_values(self, payloads: list[Tensor], ablate_values: bool = False):
        r = self.config.memory
        values = [v[None] for v in payloads]
        if ablate_values:
            values = [torch.zeros_like(v) for v in values]
        if self.training:
            values = [storage_noise(v, noise_std=r.noise_std, quantization_step=r.quantization_step)
                      for v in values]
        return values, [v.new_ones(v.shape[:2]) for v in values]

    def _compact_values(self, value: Tensor, weights: Tensor):
        r = self.config.memory
        return compact_view(value, weights, method=r.compaction, compactor=self.compactor,
                            grouping=r.compaction_grouping, group_size=r.compaction_group_size,
                            memberships=r.field_memberships)

    def read_tokens(self, payloads: list[Tensor], query: Tensor, *, compact: bool = False,
                    ablate_values: bool = False) -> tuple[Tensor, Tensor]:
        values, weights = self._prepare_values(payloads, ablate_values)
        auxiliary = query.sum() * 0
        if isinstance(self.reader, MultiSpaceReader):
            return self.reader(values, query, weights), auxiliary
        if compact and values[0].shape[1] > 0:
            view = self._compact_values(values[0], weights[0])
            auxiliary = contribution_loss(self.reader, values[0], weights[0], view.values, view.weights, query)
            return self.reader(view.values, query, view.weights).tokens, auxiliary
        if self.training and self.config.memory.merge_loss_weight and values[0].shape[1] > 1:
            r = self.config.memory
            view = compact_view(values[0], weights[0], grouping="local", group_size=r.compaction_group_size)
            auxiliary = contribution_loss(self.reader, values[0], weights[0], view.values, view.weights, query)
        return self.reader(values[0], query, weights[0]).tokens, auxiliary

    def read_pair(self, payloads: list[Tensor], query: Tensor):
        """Raw and compact paths see exactly the same noisy values and selection."""
        if isinstance(self.reader, MultiSpaceReader):
            raise ValueError("Paired compaction currently requires one space")
        values, weights = self._prepare_values(payloads)
        raw = self.reader(values[0], query, weights[0]).tokens
        view = self._compact_values(values[0], weights[0])
        compact = self.reader(view.values, query, view.weights).tokens
        auxiliary = contribution_loss(self.reader, values[0], weights[0], view.values, view.weights, query)
        return raw, compact, auxiliary

    def shared_compute_tokens(self, prompt: Tensor) -> Tensor:
        """Same query/reader/soft-slot path, no external information or payload reads."""
        query = self.query(prompt)
        # Several records prevent a null-read shortcut while carrying no source data.
        payloads = [query.new_zeros(max(1, n), d) for n, d in
                    zip(self.config.memory.neighbors, self.config.memory.payload_dims, strict=True)]
        return self.read_tokens(payloads, query)[0]

    def _context(self, prompt: Tensor, memory: Tensor | None) -> Tensor:
        context = self.backbone.embed(prompt)
        if memory is not None and memory.shape[1] > 0:
            scale = context.detach().float().square().mean().sqrt().clamp_min(1e-3)
            tokens = self.memory_norm(memory) * scale.to(memory.dtype) * self.memory_gate.sigmoid()
            context = torch.cat((context, tokens.to(context.dtype)), 1)
        return context

    def conditioned_logits(self, prompt: Tensor, target: Tensor, memory: Tensor | None) -> Tensor:
        context = self._context(prompt, memory)
        embeddings = torch.cat((context, self.backbone.embed(target[:, :-1])), 1)
        hidden = self.backbone.hidden(embeddings, torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long))
        return self.backbone.logits(hidden[:, context.shape[1] - 1:]).float()

    def conditioned_nll(self, prompt: Tensor, target: Tensor, memory: Tensor | None,
                        *, reduction: str = "mean") -> Tensor:
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("Invalid NLL reduction")
        logits = self.conditioned_logits(prompt, target, memory)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction=reduction)

    def target_ids(self, answer: str) -> Tensor:
        ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        if len(ids) > self.config.train.max_target_tokens:
            raise ValueError(f"Target uses {len(ids)} tokens; limit is {self.config.train.max_target_tokens}. Do not silently truncate complete targets.")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def forward(self, prompt: Tensor, target: Tensor, records: list[tuple[Tensor, ...]],
                required: list[int], *, step: int = 0, arm: str | None = None,
                compact: bool = False, ablate_values: bool = False) -> ForwardResult:
        t, r = self.config.train, self.config.memory
        arm = arm or t.arm
        zero = self.memory_gate * 0
        if arm in {"no_memory", "oracle_text"}:
            nll = self.conditioned_nll(prompt, target, None)
            return ForwardResult(nll, nll, zero, zero, [])
        if arm == "shared_compute":
            nll = self.conditioned_nll(prompt, target, self.shared_compute_tokens(prompt))
            return ForwardResult(nll, nll, zero, zero, [], raw_nll=nll)
        if r.read_steps > 1:
            return self.forward_multiread(prompt, target, records, required, step=step)
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
        if compact and present and r.compaction_objective == "paired":
            raw, compact_memory, auxiliary = self.read_pair(payloads, q)
            raw_logits = self.conditioned_logits(prompt, target, raw)
            compact_logits = self.conditioned_logits(prompt, target, compact_memory)
            raw_nll = F.cross_entropy(raw_logits.reshape(-1, raw_logits.shape[-1]), target.reshape(-1))
            compact_nll = F.cross_entropy(compact_logits.reshape(-1, compact_logits.shape[-1]), target.reshape(-1))
            # Forward KL of raw teacher to compact student, averaged per target position.
            behavior = F.kl_div(compact_logits.log_softmax(-1), raw_logits.detach().softmax(-1),
                                reduction="none").sum(-1).mean()
            loss = (raw_nll + r.compact_task_weight * compact_nll +
                    r.compaction_loss_weight * auxiliary + r.behavior_kl_weight * behavior +
                    t.routing_weight * routing)
            return ForwardResult(loss, raw_nll, routing, auxiliary, selected,
                                 raw_nll, compact_nll, behavior)
        if arm == "direct_latent":
            memory = payloads[0].reshape(1, -1, self.width) if present else None
            auxiliary = zero
        elif present:
            memory, auxiliary = self.read_tokens(payloads, q, compact=compact, ablate_values=ablate_values)
        else:
            memory, auxiliary = None, zero
        nll = self.conditioned_nll(prompt, target, memory)
        weight = r.compaction_loss_weight if compact else r.merge_loss_weight
        loss = nll + t.routing_weight * routing + weight * auxiliary
        return ForwardResult(loss, nll, routing, auxiliary, selected)

    def forward_multiread(self, prompt: Tensor, target: Tensor,
                          records: list[tuple[Tensor, ...]], required: list[int], *, step: int = 0):
        """Scheduled read rounds; follow-up keys depend on already received evidence.

        Recompose the cumulative selected set into fixed working slots. No duplicate
        evidence and no producer-graph truncation. Discrete choices are captured by
        selected IDs; query/reader paths through previous rounds remain differentiable.
        """
        r, t = self.config.memory, self.config.train
        zero = self.memory_gate * 0
        selected = [[] for _ in r.payload_dims]
        memory, routing, reads = None, zero, 0
        oracle = t.retrieval == "oracle" or (self.training and step < t.routing_warmup)
        for _ in range(r.read_steps):
            q = self.query(prompt, memory)
            payloads, progressed = [], False
            for space, dim in enumerate(r.payload_dims):
                candidates = [i for i in range(len(records)) if i not in selected[space]]
                if candidates:
                    keys = torch.cat([records[i][2 * space] for i in candidates], 0)
                    scores = cosine_scores(self.query_maps[space](q), keys)[0]
                    remaining = [candidates.index(i) for i in required if i in candidates]
                    if self.training and t.retrieval == "learned" and remaining:
                        routing = routing + group_plan_loss(scores, [tuple(remaining)]) / len(r.payload_dims)
                    chosen = (remaining[:r.read_top_k] if oracle else
                              scores.argsort(descending=True, stable=True)[:r.read_top_k].tolist())
                    selected[space].extend(candidates[i] for i in chosen)
                    progressed = progressed or bool(chosen)
                values = ([records[i][2 * space + 1] for i in selected[space]])
                payloads.append(torch.cat(values, 0) if values else q.new_empty(0, dim))
            if not progressed:
                break
            reads += 1
            memory, _ = self.read_tokens(payloads, q)
        nll = self.conditioned_nll(prompt, target, memory)
        routing = routing / max(1, reads)
        return ForwardResult(nll + t.routing_weight * routing, nll, routing, zero,
                             selected, raw_nll=nll, read_count=reads)

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
        return self.generate_from_memory(prompt, memory, max_new_tokens)

    @torch.no_grad()
    def generate_from_memory(self, prompt: Tensor, memory: Tensor | None, max_new_tokens: int = 24) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
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
