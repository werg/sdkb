"""Discrete candidate plans and group-aware training within those candidates."""
from __future__ import annotations

import itertools
import torch
from torch import Tensor
from torch.nn import functional as F


def cosine_scores(query: Tensor, keys: Tensor, temperature: float = 0.1) -> Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    return F.normalize(query.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).T / temperature


def group_plan_loss(scores: Tensor, sufficient_groups: list[tuple[int, ...]]) -> Tensor:
    """Negative log probability of a sufficient unordered prefix under PL sampling.

    Groups must have equal cardinality and be distinct. Enumerates <=4-member
    groups. This rewards useful starters even when their isolated utility is zero.
    Hard top-k candidate discovery itself is NOT made differentiable by this loss.
    """
    if scores.ndim != 1 or not sufficient_groups:
        raise ValueError("Expected 1D candidate scores and at least one sufficient group")
    groups = {tuple(sorted(g)) for g in sufficient_groups}
    lengths = {len(g) for g in groups}
    if len(lengths) != 1 or not 1 <= next(iter(lengths)) <= 4:
        raise ValueError("Reference requires equal-size groups with 1..4 members")
    if any(len(g) != len(set(g)) or min(g) < 0 or max(g) >= scores.numel() for g in groups):
        raise ValueError("Invalid support group")
    probabilities = []
    for group in groups:
        for order in itertools.permutations(group):
            available = torch.ones_like(scores, dtype=torch.bool)
            log_probability = scores.sum() * 0
            for index in order:
                log_probability = log_probability + scores[index] - torch.logsumexp(
                    scores.masked_fill(~available, -torch.inf), dim=0)
                available = available.clone()
                available[index] = False
            probabilities.append(log_probability)
    return -torch.logsumexp(torch.stack(probabilities), dim=0)


def utility_ranking_loss(scores: Tensor, utilities: Tensor, tolerance: float = 1e-6) -> Tensor:
    """Do not interpret indistinguishable isolated utilities as anti-group evidence."""
    if scores.shape != utilities.shape:
        raise ValueError("Shape mismatch")
    if (utilities.max() - utilities.min()).detach().item() <= tolerance:
        return scores.sum() * 0
    return -(utilities.detach().softmax(-1) * scores.log_softmax(-1)).sum()


def complete_support_recall(selected: set[str], groups: list[set[str]]) -> float:
    return float(any(group <= selected for group in groups))


def validate_routing_dataset(config, episodes) -> None:
    """Do not turn supplied teacher context into fictitious sufficient-set labels."""
    if config.train.retrieval != 'learned' or config.train.arm not in {'memory', 'direct_latent'}:
        return
    informative = 0
    for episode in episodes:
        if episode.support_annotation != 'verified':
            raise ValueError('Learned group routing requires verified support labels; provided_context '
                             'episodes support oracle training only. Collect utility/sufficiency labels first.')
        if not 1 <= len(episode.required_ids) <= 4:
            raise ValueError('Learned group routing requires 1..4 verified required records')
        informative += len(episode.supports) > len(episode.required_ids)
    if not informative and not config.train.bank_dir:
        raise ValueError('Learned routing has no competing candidates: every candidate is required. '
                         'Provide causally eligible distractors or use oracle retrieval.')
