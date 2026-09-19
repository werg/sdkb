"""Stored-only causal read sessions. No source trajectory or producer is accepted."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
from torch import Tensor

from .store import DiskStore, ReadPlan, Selection
from .recurrence import LoopMemory


@dataclass
class ReadSession:
    memory: Tensor | LoopMemory | None
    plans: list[list[ReadPlan]]
    selected_ids: list[list[str]]
    query_keys: list[Tensor]
    payload_accounting: list[dict] = field(default_factory=list)


@torch.no_grad()
def read_session(agent, store: DiskStore, prompt: Tensor, *, namespace: str,
                 generation: str, query_time: int, domain: str = "research",
                 oracle_ids: tuple[str, ...] | None = None,
                 ablate_values: bool = False, compact: bool = False,
                 exclude_ids: frozenset[str] = frozenset(), cluster_bank=None,
                 fixed_plans: list[list[ReadPlan]] | None = None) -> ReadSession:
    """Gather a captured read plan at each safe boundary, then update working slots.

    Learned retrieval searches the whole selected namespace and excludes previous
    IDs. Oracle mode supplies a fixed evidence plan, not target-dependent subsets.
    The cumulative payloads are reread at every boundary to revalidate visibility.
    This correctness path bounds *working slots*, not host payloads or I/O cost.
    """
    if agent.training:
        raise ValueError("Stored sessions require eval mode")
    r = agent.config.memory
    if fixed_plans is not None:
        validate_fixed_plans(fixed_plans, r, namespace, generation, domain, query_time, exclude_ids)
    if r.read_timing == "loop_boundary":
        if compact or cluster_bank is not None or r.stream_reads:
            raise ValueError("In-loop sessions currently read raw materialized payloads")
        return loop_read_session(agent, store, prompt, namespace=namespace, generation=generation,
                                 query_time=query_time, domain=domain, oracle_ids=oracle_ids,
                                 ablate_values=ablate_values, exclude_ids=exclude_ids, fixed_plans=fixed_plans)
    if cluster_bank is not None and (r.stream_reads or len(r.payload_dims) != 1 or compact):
        raise ValueError("Persistent codes require the single-space materialized reader, without re-compaction")
    if r.stream_reads and compact:
        raise ValueError("Streamed compaction is not implemented; use raw streaming or materialized compaction")
    selected = [[] for _ in r.payload_dims]
    memory, plans, keys, accounting = None, [], [], []
    for read in range(r.read_steps):
        if fixed_plans is not None and read >= len(fixed_plans):
            break
        q, routing_query = agent.query_pair(prompt, memory)
        keys.append(routing_query.detach().cpu())
        step_plans, progressed = [], False
        payloads = []
        for space, dim in enumerate(r.payload_dims):
            if fixed_plans is not None:
                plan = fixed_plans[read][space]
            elif oracle_ids is None:
                plan = store.search(agent.query_maps[space](routing_query)[0], namespace=namespace,
                                    space=f"s{space}", generation=generation, domain=domain,
                                    query_time=query_time,
                                    top_k=agent.requested_records(q, r.neighbors[space] if r.read_steps == 1 else r.read_top_k),
                                    exclude_ids=exclude_ids | frozenset(selected[space]))
            else:
                available = [rid for rid in oracle_ids if rid not in selected[space] and rid not in exclude_ids]
                count = len(available) if r.read_steps == 1 else r.read_top_k
                plan = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                tuple(Selection(rid, 0.) for rid in available[:count]))
            progressed = progressed or bool(plan.selections)
            selected[space].extend(s.record_id for s in plan.selections)
            step_plans.append(plan)
            cumulative = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                  tuple(Selection(rid, 0.) for rid in selected[space]))
            if not r.stream_reads and cluster_bank is None:
                values = store.fetch(cumulative)
                payloads.append(torch.stack(values).float().to(agent.device) if values else q.new_empty(0, dim))
        if not progressed:
            break
        plans.append(step_plans)
        if cluster_bank is not None:
            packed = cluster_bank.fetch(cumulative)
            value = packed.values.float().to(agent.device)
            if ablate_values:
                value = torch.zeros_like(value)
            memory = agent.reader(value, q, packed.weights.to(agent.device)).tokens
            accounting.append({"clusters": packed.used_clusters, "raw_fallback_ids": packed.raw_fallback_ids,
                               "serialized_value_bytes": packed.serialized_value_bytes,
                               "logical_tensor_bytes": packed.logical_tensor_bytes})
        elif r.stream_reads:
            from .streaming import read_stream
            def chunks(plan=cumulative):
                for values, weights in store.iter_fetch(plan, r.chunk_size):
                    yield torch.zeros_like(values) if ablate_values else values, weights
            memory = read_stream(agent.reader, q, chunks).tokens
        elif agent.config.train.arm == "direct_latent":
            memory = payloads[0].reshape(1, -1, agent.width)
            if ablate_values:
                memory = torch.zeros_like(memory)
        else:
            memory, _ = agent.read_tokens(payloads, q, compact=compact, ablate_values=ablate_values)
    return ReadSession(memory, plans, selected, keys[:len(plans)], accounting)


@torch.no_grad()
def loop_read_session(agent, store: DiskStore, prompt: Tensor, *, namespace: str,
                      generation: str, query_time: int, domain: str,
                      oracle_ids: tuple[str, ...] | None, ablate_values: bool,
                      exclude_ids: frozenset[str], fixed_plans=None) -> ReadSession:
    """Prefix-only read-plan construction at actual recurrent core boundaries."""
    if agent.training:
        raise ValueError("Stored sessions require eval mode")
    r = agent.config.memory
    selected = [[] for _ in r.payload_dims]
    plans, keys, memory = [], [], None
    if agent.backbone.loops == 1:
        return ReadSession(None, [], selected, [])
    def provider(completed, query, routing_query):
        nonlocal memory
        if completed > r.read_steps or (fixed_plans is not None and completed > len(fixed_plans)):
            return memory
        step_plans, payloads, progressed = [], [], False
        for space, dim in enumerate(r.payload_dims):
            if fixed_plans is not None:
                plan = fixed_plans[completed - 1][space]
            elif oracle_ids is None:
                plan = store.search(agent.query_maps[space](routing_query)[0], namespace=namespace,
                                    space=f"s{space}", generation=generation, domain=domain,
                                    query_time=query_time,
                                    top_k=agent.requested_records(query, r.neighbors[space] if r.read_steps == 1 else r.read_top_k),
                                    exclude_ids=exclude_ids | frozenset(selected[space]))
            else:
                available = [rid for rid in oracle_ids if rid not in selected[space] and rid not in exclude_ids]
                limit = len(available) if r.read_steps == 1 else r.read_top_k
                plan = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                tuple(Selection(rid, 0.) for rid in available[:limit]))
            progressed = progressed or bool(plan.selections)
            selected[space].extend(s.record_id for s in plan.selections)
            step_plans.append(plan)
            cumulative = ReadPlan(namespace, f"s{space}", generation, domain, query_time,
                                  tuple(Selection(rid, 0.) for rid in selected[space]))
            values = store.fetch(cumulative)
            payloads.append(torch.stack(values).float().to(agent.device) if values else query.new_empty(0, dim))
        if progressed:
            plans.append(step_plans)
            keys.append(routing_query.detach().cpu())
            memory, _ = agent.read_tokens(payloads, query, ablate_values=ablate_values)
        return memory
    schedule = agent.plan_loop_memory(prompt, provider, include_routing_query=True)
    return ReadSession(schedule, plans, selected, keys)


def validate_fixed_plans(plans, memory_config, namespace, generation, domain, query_time, excluded):
    """Captured selections never grant visibility or permit changing read boundaries."""
    if len(plans) > memory_config.read_steps:
        raise ValueError('Captured plan exceeds configured read boundaries')
    seen = [set() for _ in memory_config.payload_dims]
    for step in plans:
        if len(step) != len(seen):
            raise ValueError('Captured plan has incompatible space boundaries')
        for space, plan in enumerate(step):
            if (plan.namespace, plan.space, plan.generation, plan.domain, plan.query_time) != (
                    namespace, f's{space}', generation, domain, query_time):
                raise ValueError('Captured plan crosses a namespace/version/time/authorization boundary')
            ids = [s.record_id for s in plan.selections]
            if len(ids) != len(set(ids)) or seen[space].intersection(ids) or set(ids).intersection(excluded):
                raise ValueError('Captured plan duplicates or includes excluded evidence')
            seen[space].update(ids)
