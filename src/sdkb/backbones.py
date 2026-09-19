"""Cache-free causal backbone adapters and optional tied full-stack recurrence.

LFM2 is a hybrid, not a standard attention-only decoder. We reuse its public
inputs_embeds path and reset all conv/KV state on every pass. This is an explicit
experimental conversion, not a claim that the pretrained checkpoint was looped.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ByteTokenizer:
    """Offline tests only. Production LFM uses its own tokenizer/chat template."""
    pad_token_id, bos_token_id, eos_token_id, vocab_size = 0, 1, 2, 259

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [byte + 3 for byte in text.encode("utf-8")]
        return ([1] + ids) if add_special_tokens else ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return bytes(i - 3 for i in ids if 3 <= i < 259).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: list[dict], *, tokenize: bool = False,
                            add_generation_prompt: bool = True, **_kwargs):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant: "
        return self.encode(text, add_special_tokens=True) if tokenize else text


class TinyBackbone(nn.Module):
    def __init__(self, width: int = 64, layers: int = 2, heads: int = 4,
                 max_length: int = 2048) -> None:
        super().__init__()
        self.width, self.max_length = width, max_length
        self.embedding = nn.Embedding(259, width, padding_idx=0)
        self.position = nn.Embedding(max_length, width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, dim_feedforward=width * 3,
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.gradient_checkpointing = False
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.position.weight, std=0.02)

    def embed(self, ids: Tensor) -> Tensor:
        return self.embedding(ids)

    def core(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.shape[1] > self.max_length:
            raise ValueError("Tiny backbone context limit exceeded")
        from torch.utils.checkpoint import checkpoint
        positions = torch.arange(embeddings.shape[1], device=embeddings.device)
        x = embeddings + self.position(positions)[None]
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Bind layer now; do not capture the final loop variable in a closure.
                def run(hidden, module=layer):
                    return module(hidden, src_mask=causal, src_key_padding_mask=~mask.bool())
                x = checkpoint(run, x, use_reentrant=False)
            else:
                x = layer(x, src_mask=causal, src_key_padding_mask=~mask.bool())
        return self.norm(x)

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(hidden, self.embedding.weight)


class HFBackbone(nn.Module):
    def __init__(self, model_id: str = "LiquidAI/LFM2.5-230M", *, revision: str = "main",
                 dtype: torch.dtype = torch.float32, gradient_checkpointing: bool = True,
                 attention_implementation: str = "sdpa") -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'hf' extra; core tests do not require Transformers") from exc
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, trust_remote_code=False,
            attn_implementation=attention_implementation, use_safetensors=True,
        )
        if getattr(self.lm.config, "model_type", None) not in {"lfm2", "llama"}:
            raise ValueError("This adapter is validated structurally only for lfm2 and llama")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        self.width = self.lm.config.hidden_size
        self.max_length = min(getattr(self.lm.config, "max_position_embeddings", 32768), 32768)
        self.lm.config.use_cache = False
        if gradient_checkpointing:
            self.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.resolved_revision = getattr(self.lm.config, "_commit_hash", None) or revision

    def embed(self, ids: Tensor) -> Tensor:
        return self.lm.get_input_embeddings()(ids)

    def core(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.shape[1] > self.max_length:
            raise ValueError("Configured context limit exceeded")
        return self.lm.base_model(inputs_embeds=embeddings, attention_mask=mask,
                                  use_cache=False, return_dict=True).last_hidden_state

    def logits(self, hidden: Tensor) -> Tensor:
        return self.lm.get_output_embeddings()(hidden)


class RecurrentBackbone(nn.Module):
    """Reuse the entire pretrained stack with a gated, causal refinement update.

    loops=1 is exactly the underlying backbone path. Zero loop gate also preserves
    its output for any loop count. Each proposal uses original embeddings plus a
    normalized same-position previous state. No future-token pooling or cache reuse.
    Weight sharing saves parameters, NOT FLOPs, activations, or decoding latency.
    """
    def __init__(self, core: nn.Module, loops: int = 1) -> None:
        super().__init__()
        if loops < 1:
            raise ValueError("At least one loop is required")
        self.base, self.loops, self.width = core, loops, core.width
        self.loop_gate = nn.Parameter(torch.zeros(()))
        self.feedback_norm = nn.LayerNorm(self.width)

    def embed(self, ids: Tensor) -> Tensor:
        return self.base.embed(ids)

    def hidden(self, embeddings: Tensor, mask: Tensor, loops: int | None = None) -> Tensor:
        count = self.loops if loops is None else loops
        if count < 1:
            raise ValueError("At least one loop is required")
        hidden = self.base.core(embeddings, mask)
        scale = embeddings.detach().float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
        for _ in range(1, count):
            feedback = self.feedback_norm(hidden) * scale.to(hidden.dtype)
            proposal = self.base.core(embeddings + feedback.to(embeddings.dtype), mask)
            hidden = hidden + self.loop_gate.tanh() * (proposal - hidden)
        return hidden

    def logits(self, hidden: Tensor) -> Tensor:
        return self.base.logits(hidden)
