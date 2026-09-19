"""Stored-only causal read sessions. No source trajectory or producer is accepted."""
from __future__ import annotations

from dataclasses import dataclass, field
import torch
from torch import Tensor

from .store import DiskStore, ReadPlan, Selection


@dataclass
class ReadSession:
    memory: Tensor | None
    plans: list[list[ReadPlan]]
    selected_ids: list[list[str]]
    query_keys: list[Tensor]
    payload_accounting: list[dict] = field(default_factory=list)


@torch.no_grad()
def read_session(agent, store: DiskStore, prompt: Tensor, *, namespace: str,
                 generation: str, query_time: int, domain: str = "research",
                 oracle_ids: tuple[str, ...] | None = None,
                 ablate_values: bool = False, compact: bool = False,
                 exclude_ids: frozenset[str] = frozenset(), cluster_bank=None) -> ReadSession:
    """Gather a captured read plan at each safe boundary, then update working slots.

    Learned retrieval searches the whole selected namespace and excludes previous
    IDs. Oracle mode supplies a fixed evidence plan, not target-dependent subsets.
    The cumulative payloads are reread at every boundary to revalidate visibility.
    This correctness path bounds *working slots*, not host payloads or I/O cost.
    """
    if agent.training:
        raise ValueError("Stored sessions require eval mode")
    r = agent.config.memory
    if cluster_bank is not None and (r.stream_reads or len(r.payload_dims) != 1 or compact):
        raise ValueError("Persistent codes require the single-space materialized reader, without re-compaction")
    if r.stream_reads and compact:
        raise ValueError("Streamed compaction is not implemented; use raw streaming or materialized compaction")
    selected = [[] for _ in r.payload_dims]
    memory, plans, keys, accounting = None, [], [], []
    for read in range(r.read_steps):
        q = agent.query(prompt, memory)
        keys.append(q.detach().cpu())
        step_plans, progressed = [], False
        payloads = []
        for space, dim in enumerate(r.payload_dims):
            if oracle_ids is None:
                plan = store.search(agent.query_maps[space](q)[0], namespace=namespace,
                                    space=f"s{space}", generation=generation, domain=domain,
                                    query_time=query_time,
                                    top_k=r.neighbors[space] if r.read_steps == 1 else r.read_top_k,
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
