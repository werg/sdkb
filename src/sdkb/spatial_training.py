"""Whole-trajectory recurrent training from an immutable stored latent bank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .agent import SDKBAgent
from .key_index import PublishedKeyIndex
from .recurrence import SpatialReadSite
from .routing import cosine_scores, group_plan_loss
from .spatial_data import validate_spatial_row
from .store import DiskStore, ReadPlan, Selection


@dataclass
class SpatialForwardResult:
    loss: Tensor
    nll: Tensor
    routing: Tensor
    metrics: dict[str, Any]


def spatial_bank_forward(agent: SDKBAgent, store: DiskStore, index: PublishedKeyIndex,
                         rows: list[dict[str, Any]], *, limits: tuple[int, ...],
                         routing_candidates: int, pad_token_id: int = 0) -> SpatialForwardResult:
    """Train many causal sites; same-level bank searches execute as one matrix scan.

    Supplied verified positives keep the reader useful while the address projections
    learn against global candidates. Only immutable stored payloads enter the read path.
    """
    if not rows or routing_candidates < 1:
        raise ValueError("Spatial bank training needs rows and routing candidates")
    for row in rows:
        validate_spatial_row(row)
    memory = agent.config.memory
    if (len(limits) != len(memory.payload_dims)
            or any(limit < 1 or limit > cap
                   for limit, cap in zip(limits, memory.neighbors, strict=True))):
        raise ValueError("Spatial limits must fit every configured memory space")
    if any(row["read_slots"] != memory.read_slots for row in rows):
        raise ValueError("Packed and configured spatial slot counts differ")
    site_count = len(rows[0]["sites"])
    levels = tuple(site["level"] for site in rows[0]["sites"])
    if any(len(row["sites"]) != site_count
           or tuple(site["level"] for site in row["sites"]) != levels for row in rows):
        raise ValueError("A spatial batch needs one shared site and level layout")
    if any(len(site["required_ids"]) > min(limits)
           for row in rows for site in row["sites"]):
        raise ValueError("Every space must have room for all verified positives")

    batch = len(rows)
    maximum = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((batch, maximum), pad_token_id, dtype=torch.long,
                           device=agent.device)
    labels = torch.full_like(input_ids, -100)
    attention = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rows):
        length = len(row["input_ids"])
        input_ids[row_index, :length] = torch.tensor(row["input_ids"], device=agent.device)
        labels[row_index, :length] = torch.tensor(row["labels"], device=agent.device)
        attention[row_index, :length] = 1
    sites = tuple(SpatialReadSite(
        torch.tensor([row["sites"][site]["query_position"] for row in rows],
                     dtype=torch.long, device=agent.device),
        torch.tensor([row["sites"][site]["workspace_start"] for row in rows],
                     dtype=torch.long, device=agent.device),
        levels[site],
    ) for site in range(site_count))
    site_indices = {level: tuple(index for index, value in enumerate(levels) if value == level)
                    for level in set(levels)}
    routing_terms: list[Tensor] = []
    selected_counts = [0] * len(memory.payload_dims)
    learned_hits = [0] * len(memory.payload_dims)
    read_sites = 0

    def provider(level, active, query, routing_query):
        nonlocal read_sites
        indices = site_indices[level]
        if len(active) != len(indices):
            raise ValueError("Spatial provider site order changed")
        metadata = [rows[row]["sites"][site]
                    for site in indices for row in range(batch)]
        payloads, weights = [], []
        for space, (width, limit) in enumerate(
                zip(memory.payload_dims, limits, strict=True)):
            name = f"s{space}"
            address = agent.query_maps[space](routing_query)
            plans = index.search_batch(
                address, top_k=max(limit, routing_candidates),
                namespace=index.namespace, space=name, generation=index.generation,
                domains=tuple(item["domain"] for item in metadata),
                query_times=tuple(item["query_time"] for item in metadata),
            )
            chosen_plans = []
            for row_index, (plan, item) in enumerate(zip(plans, metadata, strict=True)):
                found = [selection.record_id for selection in plan.selections]
                required = list(item["required_ids"])
                candidate_ids = list(dict.fromkeys(found + required))
                keys = index.keys_for_ids(name, candidate_ids, domain=item["domain"],
                                          query_time=item["query_time"]).to(agent.device)
                scores = cosine_scores(address[row_index:row_index + 1], keys)[0]
                positive_indices = tuple(candidate_ids.index(record_id)
                                         for record_id in required)
                routing_terms.append(group_plan_loss(scores, [positive_indices]))
                chosen = required + [record_id for record_id in found
                                     if record_id not in required]
                chosen = chosen[:limit]
                learned_hits[space] += int(set(required) <= set(found[:limit]))
                selected_counts[space] += len(chosen)
                chosen_plans.append(ReadPlan(
                    index.namespace, name, index.generation, item["domain"],
                    item["query_time"], tuple(Selection(record_id, 0.0)
                                              for record_id in chosen),
                ))
            values = store.fetch_many(chosen_plans)
            count = max(map(len, values))
            device_rows = [torch.stack([value.to(agent.device).float() for value in row])
                           for row in values]
            if any(row.shape[1] != width for row in device_rows):
                raise ValueError("Stored spatial payload width differs from configuration")
            payloads.append(torch.stack([
                F.pad(row, (0, 0, 0, count - row.shape[0])) for row in device_rows]))
            weights.append(torch.stack([
                torch.cat((query.new_ones(row.shape[0]),
                           query.new_zeros(count - row.shape[0]))) for row in device_rows]))
        read_sites += len(metadata)
        return agent._read_padded_batch(payloads, weights, query)

    hidden = agent.spatial_recurrent_hidden(input_ids, attention, sites, provider)
    logits = agent.backbone.logits(hidden).float()
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    if not routing_terms or read_sites != batch * site_count:
        raise ValueError("Every spatial site must execute exactly one bank read")
    routing = torch.stack(routing_terms).mean()
    loss = nll + agent.config.train.routing_weight * routing
    denominator = read_sites
    metrics = {
        "read_sites": read_sites,
        "selected_counts": [count / denominator for count in selected_counts],
        "learned_positive_recall": [hits / denominator for hits in learned_hits],
        "selected_payload_bytes": sum(count * width * 2 for count, width in
                                      zip(selected_counts, memory.payload_dims, strict=True))
                                  / batch,
        "supervised_tokens": int((labels >= 0).sum()),
    }
    return SpatialForwardResult(loss, nll, routing, metrics)
