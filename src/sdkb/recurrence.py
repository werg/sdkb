"""Identity-at-one-loop conversion with a weight-shared middle block.

No copied decoder layers, no cross-loop KV/conv cache, and no position increments
along depth. All native pretrained normalization remains at its original location.
See docs/recurrence.md for the research basis and the limits of this reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LoopWrite:
    """A causally located fixed-size soft-memory result, not a sequence append."""
    start: int | Tensor
    tokens: Tensor


@dataclass(frozen=True)
class LoopMemory:
    """Captured prefix-only reads, replayable for any continuation of that prefix.

    Entry zero is applied between core passes 1 and 2. None means no available
    result. After the final read its value is re-injected, not retrieved again.
    """
    prefix_length: int
    slots: int
    loops: int
    events: tuple[Tensor | None, ...]

    def __post_init__(self):
        if self.loops < 1 or self.slots < 1 or self.prefix_length < 1:
            raise ValueError("Invalid loop-memory layout")
        if len(self.events) != self.loops - 1:
            raise ValueError("One captured result per recurrent boundary is required")
        for event in self.events:
            if event is not None and (event.ndim != 3 or event.shape[1] != self.slots):
                raise ValueError("Invalid captured soft-memory shape")

    def callback(self, completed: int, _state: Tensor, _anchor: Tensor) -> LoopWrite | None:
        event = self.events[completed - 1]
        return None if event is None else LoopWrite(self.prefix_length, event)


def _logit(value: float) -> Tensor:
    if not 0 < value < 1:
        raise ValueError("A live recurrent mixing coefficient must be strictly in (0, 1)")
    return torch.logit(torch.tensor(value, dtype=torch.float32))


class AnchoredBridge(nn.Module):
    """Conservative, live re-entry; normalization and scale are strictly tokenwise.

    u = p + sigmoid(a) * RMS(p) * W([RMSNorm(h), RMSNorm(p)]).
    W starts as [I, -I]. Thus similar states require only a small correction, while
    the persistent prelude remains an anchor. No gate or projection is dead at init.
    This is an SDKB adaptation, not a verbatim implementation of a cited paper.
    """
    def __init__(self, width: int, input_mix: float, update_mix: float):
        super().__init__()
        self.state_norm = nn.RMSNorm(width, eps=1e-5)
        self.anchor_norm = nn.RMSNorm(width, eps=1e-5)
        self.reentry = nn.Linear(2 * width, width, bias=False)
        self.input_logit = nn.Parameter(_logit(input_mix))
        self.update_logit = nn.Parameter(_logit(update_mix))
        self.memory_norm = nn.RMSNorm(width, eps=1e-5)
        self.memory_projection = nn.Linear(width, width, bias=False)
        self.memory_logit = nn.Parameter(_logit(input_mix))
        with torch.no_grad():
            eye = torch.eye(width)
            self.reentry.weight.copy_(torch.cat((eye, -eye), dim=1))
            self.memory_projection.weight.copy_(eye)

    @staticmethod
    def scale(anchor: Tensor) -> Tensor:
        # Do not pool across time: teacher-forced targets must never set prefix scale.
        return anchor.detach().float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-5)

    def forward(self, state: Tensor, anchor: Tensor) -> Tensor:
        features = torch.cat((self.state_norm(state), self.anchor_norm(anchor)), -1)
        delta = self.reentry(features).to(anchor.dtype)
        return anchor + self.input_logit.sigmoid().to(anchor.dtype) * self.scale(anchor).to(anchor.dtype) * delta

    def inject(self, inputs: Tensor, anchor: Tensor, write: LoopWrite) -> Tensor:
        start, tokens = write.start, write.tokens
        if tokens.ndim != 3:
            raise ValueError("Soft result must be a batch of token vectors")
        if tokens.shape[0] != inputs.shape[0] or tokens.shape[-1] != inputs.shape[-1]:
            raise ValueError("Soft-result batch/width mismatch")
        if isinstance(start, Tensor):
            if start.shape != (inputs.shape[0],) or start.dtype not in (torch.int32, torch.int64):
                raise ValueError("Batched soft-result starts must be one integer per row")
            indices = start[:, None] + torch.arange(tokens.shape[1], device=inputs.device)[None]
            if bool((indices < 0).any()) or bool((indices >= inputs.shape[1]).any()):
                raise ValueError("Soft result lies outside reserved memory slots")
            gather = indices[..., None].expand(-1, -1, inputs.shape[-1])
            anchors = anchor.gather(1, gather)
            delta = self.memory_projection(self.memory_norm(tokens)).to(inputs.dtype)
            delta = delta * self.scale(anchors).to(inputs.dtype)
            update = torch.zeros_like(inputs).scatter(1, gather, delta)
            return inputs + self.memory_logit.sigmoid().to(inputs.dtype) * update
        stop = start + tokens.shape[1]
        if start < 0 or stop > inputs.shape[1]:
            raise ValueError("Soft result lies outside reserved memory slots")
        delta = self.memory_projection(self.memory_norm(tokens)).to(inputs.dtype)
        delta = delta * self.scale(anchor[:, start:stop]).to(inputs.dtype)
        middle = inputs[:, start:stop] + self.memory_logit.sigmoid().to(inputs.dtype) * delta
        return torch.cat((inputs[:, :start], middle, inputs[:, stop:]), 1)

    def update(self, state: Tensor, proposal: Tensor) -> Tensor:
        return state + self.update_logit.sigmoid().to(state.dtype) * (proposal - state)


class MiddleBlockBackbone(nn.Module):
    """Prelude[0:a], shared core[a:b] repeated R times, coda[b:L].

    The parent retains ownership of every decoder layer. Merely changing R never
    changes parameter count or parameter identity. R=1 bypasses every addition.
    Boundary callbacks must be pure under recomputation; native layer checkpointing
    never re-executes a retrieval callback.
    """
    def __init__(self, base: nn.Module, loops: int = 2, *, start: int, end: int,
                 input_mix: float = 0.1, update_mix: float = 0.1):
        super().__init__()
        self.base, self.width, self.loops = base, base.width, loops
        self.start, self.end = start, end
        if not 0 <= start < end <= base.layer_count or loops < 1:
            raise ValueError("Invalid prelude/core/coda partition or depth")
        self.bridge = AnchoredBridge(self.width, input_mix, update_mix)

    def embed(self, ids: Tensor) -> Tensor:
        return self.base.embed(ids)

    def hidden(self, embeddings: Tensor, mask: Tensor, loops: int | None = None, *,
               boundary: Callable[[int, Tensor, Tensor], LoopWrite | None] | None = None,
               plan_only: bool = False, trace: list[Tensor] | None = None) -> Tensor:
        count = self.loops if loops is None else loops
        if count < 1:
            raise ValueError("At least one core pass is required")
        if plan_only and boundary is None:
            raise ValueError("Planning requires a read callback")
        hidden, context = self.base.prepare_layers(embeddings, mask)
        anchor = self.base.run_layers(hidden, context, 0, self.start)
        state = self.base.run_layers(anchor, context, self.start, self.end)
        if trace is not None:
            trace.append(state)
        for completed in range(1, count):
            write = boundary(completed, state, anchor) if boundary is not None else None
            # A prefix-only read plan does not need an unused last proposal or coda.
            if plan_only and completed == count - 1:
                return state
            inputs = self.bridge(state, anchor)
            if write is not None:
                inputs = self.bridge.inject(inputs, anchor, write)
            proposal = self.base.run_layers(inputs, context, self.start, self.end)
            state = self.bridge.update(state, proposal)
            if trace is not None:
                trace.append(state)
        if plan_only:
            return state
        return self.base.finish_layers(self.base.run_layers(state, context, self.end, self.base.layer_count))

    def logits(self, hidden: Tensor) -> Tensor:
        return self.base.logits(hidden)

    def manifest(self, loops: int | None = None) -> dict:
        count = self.loops if loops is None else loops
        return {"mode": "middle_block", "prelude": [0, self.start],
                "core": [self.start, self.end], "coda": [self.end, self.base.layer_count],
                "loops": count, "layer_visits": self.base.layer_count + (count - 1) * (self.end - self.start),
                "layer_types": self.base.layer_types,
                "new_bridge_parameters": sum(p.numel() for p in self.bridge.parameters()),
                "input_mix": float(self.bridge.input_logit.detach().sigmoid()),
                "update_mix": float(self.bridge.update_logit.detach().sigmoid()),
                "memory_mix": float(self.bridge.memory_logit.detach().sigmoid()),
                "cache": "disabled; no KV or convolution state crosses a depth boundary"}
