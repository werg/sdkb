"""Fixed-plan, stored-payload training reads from a published frozen bank."""
from __future__ import annotations

import torch

from .agent import ForwardResult, SDKBAgent
from .data import Episode
from .routing import cosine_scores, group_plan_loss
from .store import DiskStore, ReadPlan, Selection, lookup_record


def stored_corpus_forward(agent: SDKBAgent, store: DiskStore, episode: Episode, *,
                          generation: str, limits: tuple[int, ...],
                          namespace: str = 'corpus') -> tuple[ForwardResult, dict]:
    """Plan each space from a causal native prefix, then train on fetched values.

    Known sufficient supports are delivered during this initial mixed-selection
    stage; retrieval is trained against the global candidates in the same read.
    No source text reaches the writer or the consumer prompt on this path.
    """
    if (agent.config.memory.read_timing != 'loop_boundary' or agent.backbone.loops != 2 or
            len(limits) != len(agent.config.memory.payload_dims) or
            any(k < len(episode.required_ids) or k > cap for k, cap in
                zip(limits, agent.config.memory.neighbors, strict=True))):
        raise ValueError('Stored corpus training requires one native read and valid per-space limits')
    if episode.support_annotation != 'verified' or not 1 <= len(episode.required_ids) <= 4:
        raise ValueError('Corpus routing needs 1..4 verified sufficient supports')
    if any(source.created_at >= episode.query_time for source in episode.supports):
        raise ValueError('Corpus sources must precede their query')
    domain = episode.provenance.get('domain', 'research')
    prompt, target = agent.prompt_ids(episode.query), agent.target_ids(episode.answer)
    routing_terms = []
    selected_ids, learned_recalls = [], []

    def provider(completed, query, routing_query):
        if completed != 1:
            return None
        payloads = []
        for space, (dim, limit) in enumerate(zip(agent.config.memory.payload_dims, limits, strict=True)):
            space_name = f's{space}'
            address = agent.query_maps[space](routing_query)
            candidates = store.search(address[0], top_k=max(limit, 8), namespace=namespace,
                                      space=space_name, generation=generation, domain=domain,
                                      query_time=episode.query_time)
            found = [selection.record_id for selection in candidates.selections]
            # Every candidate, including supplied positives, is fetched under
            # the same immutable generation, authorization and time scope.
            candidate_ids = list(dict.fromkeys(found + list(episode.required_ids)))
            keys = torch.stack([lookup_record(store, record_id, namespace=namespace,
                                space=space_name, generation=generation, domain=domain,
                                query_time=episode.query_time).key for record_id in candidate_ids]).to(agent.device)
            scores = cosine_scores(address, keys)[0]
            positive_indices = tuple(candidate_ids.index(record_id) for record_id in episode.required_ids)
            routing_terms.append(group_plan_loss(scores, [positive_indices]))
            chosen_ids = list(episode.required_ids)
            chosen_ids.extend(record_id for record_id in found if record_id not in chosen_ids)
            chosen_ids = chosen_ids[:limit]
            selection = ReadPlan(namespace, space_name, generation, domain, episode.query_time,
                                 tuple(Selection(record_id, 0.0) for record_id in chosen_ids))
            values = store.fetch(selection)
            payloads.append(torch.stack([value.to(agent.device).float() for value in values]))
            selected_ids.append(chosen_ids)
            learned_recalls.append(float(set(episode.required_ids) <= set(found[:limit])))
            if payloads[-1].shape != (len(chosen_ids), dim):
                raise ValueError('Stored payload width or selected count changed')
        tokens, _ = agent.read_tokens(payloads, query)
        return tokens

    memory = agent.plan_loop_memory(prompt, provider, include_routing_query=True)
    nll = agent.conditioned_nll(prompt, target, memory)
    routing = torch.stack(routing_terms).mean()
    loss = nll + agent.config.train.routing_weight * routing
    result = ForwardResult(loss, nll, routing, nll * 0,
                           [list(range(len(ids))) for ids in selected_ids], raw_nll=nll,
                           read_count=1)
    return result, {'selected_ids': selected_ids,
                    'selected_counts': [len(ids) for ids in selected_ids],
                    'learned_positive_recall': learned_recalls,
                    'selected_payload_bytes': sum(len(ids) * dim * 2 for ids, dim in
                                                  zip(selected_ids, agent.config.memory.payload_dims, strict=True)),
                    'supplied_positive': True}
