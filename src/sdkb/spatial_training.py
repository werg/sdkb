"""Whole-trajectory recurrent training from an immutable stored latent bank."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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


@dataclass
class _SpatialBatchState:
    rows: list[dict[str, Any]]
    labels: Tensor
    execution: Any
    sites: tuple[SpatialReadSite, ...]
    site_indices: dict[int, tuple[int, ...]]
    routing_terms: list[Tensor]
    selected_counts: list[int]
    learned_hits: list[int]
    read_sites: int = 0


@dataclass(frozen=True)
class _ReadWave:
    active: tuple[SpatialReadSite, ...]
    query: Tensor
    addresses: tuple[Tensor, ...]
    search_addresses: tuple[Tensor, ...]
    metadata: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _LoadedSpace:
    candidate_ids: tuple[tuple[str, ...], ...]
    required_ids: tuple[tuple[str, ...], ...]
    found_ids: tuple[tuple[str, ...], ...]
    values: tuple[tuple[Tensor, ...], ...]


def _validate_layout(agent: SDKBAgent, rows: list[dict[str, Any]],
                     limits: tuple[int, ...], routing_candidates: int) -> tuple[int, tuple[int, ...]]:
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
    return site_count, levels


def _begin_batch(agent: SDKBAgent, rows: list[dict[str, Any]], *,
                 limits: tuple[int, ...], routing_candidates: int,
                 pad_token_id: int) -> _SpatialBatchState:
    site_count, levels = _validate_layout(agent, rows, limits, routing_candidates)
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
    site_indices = {level: tuple(index for index, value in enumerate(levels)
                                 if value == level) for level in set(levels)}
    execution = agent.begin_spatial_recurrent(input_ids, attention, sites)
    return _SpatialBatchState(
        rows, labels, execution, sites, site_indices, [],
        [0] * len(agent.config.memory.payload_dims),
        [0] * len(agent.config.memory.payload_dims),
    )


def _issue_wave(agent: SDKBAgent, state: _SpatialBatchState) -> _ReadWave | None:
    while state.execution.recurrent.completed < state.execution.recurrent.loops:
        pending = agent.spatial_recurrent_query(state.execution)
        if pending is not None:
            active, query, routing_query = pending
            level = state.execution.recurrent.completed
            metadata = tuple(
                state.rows[row]["sites"][site]
                for site in state.site_indices[level]
                for row in range(len(state.rows))
            )
            addresses = tuple(mapping(routing_query) for mapping in agent.query_maps)
            # Synchronize only the small address matrices before handing work to
            # CPU threads. The differentiable device expressions remain retained.
            search_addresses = tuple(address.detach().float().cpu()
                                     for address in addresses)
            return _ReadWave(active, query, addresses, search_addresses, metadata)
        state.execution = agent.advance_empty_spatial_level(state.execution)
    return None


def _load_wave(store: DiskStore, index: PublishedKeyIndex, wave: _ReadWave, *,
               limits: tuple[int, ...], routing_candidates: int) -> tuple[_LoadedSpace, ...]:
    """Perform selection and serialized payload reads without touching the GPU graph."""
    loaded = []
    for space, limit in enumerate(limits):
        name = f"s{space}"
        plans = index.search_batch(
            wave.search_addresses[space],
            top_k=max(limit, routing_candidates), namespace=index.namespace,
            space=name, generation=index.generation,
            domains=tuple(item["domain"] for item in wave.metadata),
            query_times=tuple(item["query_time"] for item in wave.metadata),
        )
        candidates, required_rows, found_rows, chosen_plans = [], [], [], []
        for plan, item in zip(plans, wave.metadata, strict=True):
            found = tuple(selection.record_id for selection in plan.selections)
            required = tuple(item["required_ids"])
            candidate_ids = tuple(dict.fromkeys(found + required))
            chosen = (required + tuple(record_id for record_id in found
                                       if record_id not in required))[:limit]
            candidates.append(candidate_ids)
            required_rows.append(required)
            found_rows.append(found)
            chosen_plans.append(ReadPlan(
                index.namespace, name, index.generation, item["domain"],
                item["query_time"], tuple(Selection(record_id, 0.0)
                                          for record_id in chosen),
            ))
        values = store.fetch_many(chosen_plans)
        loaded.append(_LoadedSpace(
            tuple(candidates), tuple(required_rows), tuple(found_rows),
            tuple(tuple(row) for row in values),
        ))
    return tuple(loaded)


def _consume_wave(agent: SDKBAgent, index: PublishedKeyIndex, state: _SpatialBatchState,
                  wave: _ReadWave, loaded: tuple[_LoadedSpace, ...], *,
                  limits: tuple[int, ...]) -> None:
    memory = agent.config.memory
    payloads, weights = [], []
    for space, (width, limit, result) in enumerate(
            zip(memory.payload_dims, limits, loaded, strict=True)):
        address = wave.addresses[space]
        device_rows = []
        for row_index, (candidate_ids, required, found, values, item) in enumerate(zip(
                result.candidate_ids, result.required_ids, result.found_ids,
                result.values, wave.metadata, strict=True)):
            keys = index.keys_for_ids(
                f"s{space}", candidate_ids, domain=item["domain"],
                query_time=item["query_time"],
            ).to(agent.device)
            scores = cosine_scores(address[row_index:row_index + 1], keys)[0]
            positive_indices = tuple(candidate_ids.index(record_id) for record_id in required)
            state.routing_terms.append(group_plan_loss(scores, [positive_indices]))
            state.learned_hits[space] += int(set(required) <= set(found[:limit]))
            state.selected_counts[space] += len(values)
            device_rows.append(torch.stack([
                value.to(agent.device).float() for value in values
            ]))
        count = max(map(len, result.values))
        if any(row.shape[1] != width for row in device_rows):
            raise ValueError("Stored spatial payload width differs from configuration")
        payloads.append(torch.stack([
            F.pad(row, (0, 0, 0, count - row.shape[0])) for row in device_rows
        ]))
        weights.append(torch.stack([
            torch.cat((wave.query.new_ones(row.shape[0]),
                       wave.query.new_zeros(count - row.shape[0])))
            for row in device_rows
        ]))
    state.read_sites += len(wave.metadata)
    results = agent._read_padded_batch(payloads, weights, wave.query)
    state.execution = agent.advance_spatial_recurrent(
        state.execution, wave.active, results)


def _finish_batch(agent: SDKBAgent, state: _SpatialBatchState) -> SpatialForwardResult:
    hidden = agent.finish_spatial_recurrent(state.execution)
    supervised = state.labels >= 0
    logits = agent.backbone.logits(hidden[supervised]).float()
    nll = F.cross_entropy(logits, state.labels[supervised])
    expected_reads = len(state.rows) * len(state.sites)
    if not state.routing_terms or state.read_sites != expected_reads:
        raise ValueError("Every spatial site must execute exactly one bank read")
    routing = torch.stack(state.routing_terms).mean()
    loss = nll + agent.config.train.routing_weight * routing
    denominator = state.read_sites
    metrics = {
        "read_sites": state.read_sites,
        "write_sites": sum(len(row.get("write_sites", ())) for row in state.rows),
        "selected_counts": [count / denominator for count in state.selected_counts],
        "learned_positive_recall": [hits / denominator for hits in state.learned_hits],
        "selected_payload_bytes": sum(
            count * width * 2 for count, width in
            zip(state.selected_counts, agent.config.memory.payload_dims, strict=True)
        ) / len(state.rows),
        "supervised_tokens": int(supervised.sum()),
    }
    return SpatialForwardResult(loss, nll, routing, metrics)


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
    supervised = labels >= 0
    # The tied vocabulary projection is large. Blank workspaces, prompts, tool
    # results and padding have no target, so projecting them wastes both memory and
    # compute without changing the objective.
    logits = agent.backbone.logits(hidden[supervised]).float()
    nll = F.cross_entropy(logits, labels[supervised])
    if not routing_terms or read_sites != batch * site_count:
        raise ValueError("Every spatial site must execute exactly one bank read")
    routing = torch.stack(routing_terms).mean()
    loss = nll + agent.config.train.routing_weight * routing
    denominator = read_sites
    metrics = {
        "read_sites": read_sites,
        "write_sites": sum(len(row.get("write_sites", ())) for row in rows),
        "selected_counts": [count / denominator for count in selected_counts],
        "learned_positive_recall": [hits / denominator for hits in learned_hits],
        "selected_payload_bytes": sum(count * width * 2 for count, width in
                                      zip(selected_counts, memory.payload_dims, strict=True))
                                  / batch,
        "supervised_tokens": int(supervised.sum()),
    }
    return SpatialForwardResult(loss, nll, routing, metrics)


def spatial_bank_pipeline_forward(
        agent: SDKBAgent, store: DiskStore, index: PublishedKeyIndex,
        rows: list[dict[str, Any]], *, limits: tuple[int, ...],
        routing_candidates: int, microbatch_size: int, inflight: int,
        pad_token_id: int = 0) -> SpatialForwardResult:
    """Overlap retained microbatch graphs with CPU search and serialized payload I/O.

    GPU continuations execute in deterministic round-robin order. Retrieval workers
    may finish out of order, but each captured plan remains attached to the query and
    causal metadata that produced it. The number of in-flight graphs is explicitly
    bounded, and retrieval never runs inside decoder checkpoint recomputation.
    """
    if microbatch_size < 1 or inflight < 1:
        raise ValueError("Pipeline microbatch size and in-flight count must be positive")
    _validate_layout(agent, rows, limits, routing_candidates)
    chunks = [rows[start:start + microbatch_size]
              for start in range(0, len(rows), microbatch_size)]
    if len(chunks) > 1 and inflight < 2:
        raise ValueError("Multiple pipeline microbatches require at least two in flight")
    results: list[tuple[int, SpatialForwardResult]] = []
    pending: deque[tuple[int, _SpatialBatchState, _ReadWave,
                         Future[tuple[_LoadedSpace, ...]]]] = deque()
    next_chunk = 0

    with ThreadPoolExecutor(max_workers=min(inflight, len(chunks)),
                            thread_name_prefix="sdkb-bank-read") as pool:
        def launch(index_in_step: int) -> None:
            state = _begin_batch(
                agent, chunks[index_in_step], limits=limits,
                routing_candidates=routing_candidates, pad_token_id=pad_token_id)
            wave = _issue_wave(agent, state)
            if wave is None:
                results.append((index_in_step, _finish_batch(agent, state)))
                return
            future = pool.submit(
                _load_wave, store, index, wave, limits=limits,
                routing_candidates=routing_candidates)
            pending.append((index_in_step, state, wave, future))

        while next_chunk < len(chunks) and len(pending) < inflight:
            launch(next_chunk)
            next_chunk += 1
        while pending:
            index_in_step, state, wave, future = pending.popleft()
            _consume_wave(agent, index, state, wave, future.result(), limits=limits)
            following = _issue_wave(agent, state)
            if following is None:
                results.append((index_in_step, _finish_batch(agent, state)))
                if next_chunk < len(chunks):
                    launch(next_chunk)
                    next_chunk += 1
            else:
                future = pool.submit(
                    _load_wave, store, index, following, limits=limits,
                    routing_candidates=routing_candidates)
                pending.append((index_in_step, state, following, future))

    ordered = [result for _, result in sorted(results)]
    supervised = sum(result.metrics["supervised_tokens"] for result in ordered)
    read_sites = sum(result.metrics["read_sites"] for result in ordered)
    batch = len(rows)
    nll = sum(result.nll * result.metrics["supervised_tokens"]
              for result in ordered) / supervised
    routing = sum(result.routing * result.metrics["read_sites"]
                  for result in ordered) / read_sites
    loss = nll + agent.config.train.routing_weight * routing
    spaces = len(agent.config.memory.payload_dims)
    metrics = {
        "read_sites": read_sites,
        "write_sites": sum(result.metrics["write_sites"] for result in ordered),
        "selected_counts": [
            sum(result.metrics["selected_counts"][space]
                * result.metrics["read_sites"] for result in ordered) / read_sites
            for space in range(spaces)
        ],
        "learned_positive_recall": [
            sum(result.metrics["learned_positive_recall"][space]
                * result.metrics["read_sites"] for result in ordered) / read_sites
            for space in range(spaces)
        ],
        "selected_payload_bytes": sum(
            result.metrics["selected_payload_bytes"] * len(chunk)
            for result, chunk in zip(ordered, chunks, strict=True)
        ) / batch,
        "supervised_tokens": supervised,
        "pipeline_microbatches": len(chunks),
        "pipeline_inflight": min(inflight, len(chunks)),
    }
    return SpatialForwardResult(loss, nll, routing, metrics)
