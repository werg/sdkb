"""Temporary full-cluster compaction and contribution-preserving training losses.

Persistent arbitrary-child selection is intentionally not approximated by a
full-cluster code. FullClusterCode.require_selection fails closed for subsets.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .readers import SetReader, Statistics, accumulation_dtype


def mean_and_mass(x: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    if weights.shape != x.shape[:2] or (weights < 0).any():
        raise ValueError("Expected nonnegative [B,N] weights")
    dtype = accumulation_dtype(x)
    mass = weights.to(dtype).sum(1, keepdim=True)
    mean = (x.to(dtype) * weights.to(dtype)[..., None]).sum(1, keepdim=True)
    mean = mean / mass[..., None].clamp_min(torch.finfo(dtype).tiny)
    return mean.to(x.dtype), mass.to(x.dtype)


class SyntheticCompactor(nn.Module):
    """Amortized, permutation-invariant cluster -> r synthetic ordinary records.

    Effective multiplicity is conserved. The initial output is near a replicated
    mean; shared encoder parameters, not independently optimized per-cluster codes.
    """
    def __init__(self, input_dim: int, width: int = 128, records: int = 2) -> None:
        super().__init__()
        if min(input_dim, width, records) < 1:
            raise ValueError("Dimensions must be positive")
        self.records, self.input_dim = records, input_dim
        self.encode = nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.codes = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(),
                                   nn.Linear(width, records * input_dim))
        self.multiplicity = nn.Linear(width + 1, records)
        nn.init.normal_(self.codes[-1].weight, std=1e-3)
        nn.init.zeros_(self.codes[-1].bias)

    def forward(self, x: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        mean, mass = mean_and_mass(x, weights)
        feature, _ = mean_and_mass(self.encode(x), weights)
        feature = torch.cat((feature[:, 0], torch.log1p(mass)), -1)
        codes = self.codes(feature).reshape(x.shape[0], self.records, self.input_dim)
        codes = mean + codes
        weights_out = self.multiplicity(feature).softmax(-1) * mass
        return codes, weights_out


def statistics_loss(student: Statistics, teacher: Statistics) -> Tensor:
    """Match numerator AND mass at a common log scale; teacher is detached."""
    scale = torch.maximum(student.log_scale, teacher.log_scale).detach()
    sn = student.numerator * (student.log_scale - scale).exp()
    tn = teacher.numerator.detach() * (teacher.log_scale.detach() - scale).exp()
    sm = student.mass * (student.log_scale - scale).exp()
    tm = teacher.mass.detach() * (teacher.log_scale.detach() - scale).exp()
    # Reference normalization avoids making large clusters dominate the loss.
    denominator = tm.detach().clamp_min(1.0)
    return F.mse_loss(sn / denominator, tn / denominator) + F.mse_loss(sm / denominator, tm / denominator)


def contribution_loss(reader: SetReader, raw: Tensor, raw_weights: Tensor,
                      compact: Tensor, compact_weights: Tensor, query: Tensor,
                      rollout_weight: float = 1.0) -> Tensor:
    """Teacher-state conditional statistics plus free compact rollout matching.

    The uncompressed branch is detached; the compact branch can train the reader,
    writer and/or compactor. Caller decides which parameters are frozen.
    """
    with torch.no_grad():
        teacher = reader(raw, query, raw_weights, diagnostics=True)
    loss = compact.sum() * 0
    for t, (state, target) in enumerate(zip(teacher.states, teacher.statistics, strict=True)):
        current = reader.aggregate(t, compact, query, state.detach(), compact_weights)
        loss = loss + statistics_loss(current, target) / reader.rounds
    if rollout_weight:
        predicted = reader(compact, query, compact_weights).tokens
        loss = loss + rollout_weight * F.mse_loss(predicted, teacher.tokens.detach())
    return loss


def local_merge_loss(reader: SetReader, raw: Tensor, weights: Tensor, query: Tensor) -> Tensor:
    mean, mass = mean_and_mass(raw, weights)
    return contribution_loss(reader, raw, weights, mean, mass, query)


def field_responsibilities(points: Tensor, centers: Tensor, memberships: int = 2,
                           temperature: float = 1.0) -> Tensor:
    """Two/few-nearest-center partition of unity; field shares sum to one per point."""
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("Expected [N,D] points and [C,D] centers")
    if not 1 <= memberships <= centers.shape[0] or temperature <= 0:
        raise ValueError("Invalid membership count or temperature")
    distances = torch.cdist(points.float(), centers.float()).square()
    distance, indices = distances.topk(memberships, largest=False, dim=-1)
    coefficients = (-distance / temperature).softmax(-1)
    return torch.zeros_like(distances).scatter(1, indices, coefficients).to(points.dtype)


def random_partition(n: int, group_size: int, generator: torch.Generator) -> list[list[int]]:
    """Alternative training views; a point occurs exactly once in each view."""
    if n < 0 or group_size < 1:
        raise ValueError("Invalid partition size")
    perm = torch.randperm(n, generator=generator).tolist()
    return [perm[i:i + group_size] for i in range(0, n, group_size)]


def storage_noise(x: Tensor, *, noise_std: float = 0.0, quantization_step: float = 0.0) -> Tensor:
    """Training-only perturbation; replay captures torch RNG. No inference regeneration."""
    if noise_std < 0 or quantization_step < 0:
        raise ValueError("Noise levels cannot be negative")
    if noise_std:
        x = x + noise_std * torch.randn_like(x)
    if quantization_step:
        x = x + (torch.rand_like(x) - 0.5) * quantization_step
    return x


@dataclass(frozen=True)
class FullClusterCode:
    children: tuple[str, ...]
    authorization_domain: str
    reader_version: str

    def require_selection(self, selected: tuple[str, ...], domain: str) -> None:
        if domain != self.authorization_domain:
            raise PermissionError("Compacted payload crosses an authorization boundary")
        if set(selected) != set(self.children) or len(selected) != len(self.children):
            raise ValueError("Full-cluster code cannot answer an arbitrary partial selection")
