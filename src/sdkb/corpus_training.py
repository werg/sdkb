"""Fixed-plan, stored-payload training reads from a published frozen bank."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .agent import ForwardResult, SDKBAgent
from .data import Episode
from .recurrence import LoopMemory
from .routing import cosine_scores, group_plan_loss
from .store import DiskStore, ReadPlan, Selection, lookup_record


def stored_corpus_forward(agent: SDKBAgent, store: DiskStore, episode: Episode, *,
                          generation: str, limits: tuple[int, ...],
                          namespace: str = 'corpus', searcher=None) -> tuple[ForwardResult, dict]:
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
    contrast = agent.config.train.bank_payload_contrast_weight > 0
    if contrast and len(episode.required_ids) != 1:
        raise ValueError('Stored-bank source swap currently requires one verified positive')
    searcher = searcher or store
    prompt, target = agent.prompt_ids(episode.query), agent.target_ids(episode.answer)
    routing_terms = []
    selected_ids, learned_recalls, swapped_ids = [], [], []
    swapped_tokens = None

    def provider(completed, query, routing_query):
        nonlocal swapped_tokens
        if completed != 1:
            return None
        payloads, swapped_payloads = [], []
        for space, (dim, limit) in enumerate(zip(agent.config.memory.payload_dims, limits, strict=True)):
            space_name = f's{space}'
            address = agent.query_maps[space](routing_query)
            candidates = searcher.search(address[0], top_k=max(limit + int(contrast), 8), namespace=namespace,
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
            if contrast:
                wrong_id = next((record_id for record_id in found
                                 if record_id not in chosen_ids
                                 and record_id not in episode.required_ids), None)
                if wrong_id is None:
                    raise ValueError('Stored-bank contrast needs an eligible unselected source')
                wrong = store.fetch(ReadPlan(namespace, space_name, generation, domain,
                    episode.query_time, (Selection(wrong_id, 0.0),)))[0]
                swapped = payloads[-1].clone()
                swapped[0] = wrong.to(agent.device).float()
                swapped_payloads.append(swapped)
                swapped_ids.append(wrong_id)
            selected_ids.append(chosen_ids)
            learned_recalls.append(float(set(episode.required_ids) <= set(found[:limit])))
            if payloads[-1].shape != (len(chosen_ids), dim):
                raise ValueError('Stored payload width or selected count changed')
        tokens, _ = agent.read_tokens(payloads, query)
        if contrast:
            swapped_tokens, _ = agent.read_tokens(swapped_payloads, query)
        return tokens

    memory = agent.plan_loop_memory(prompt, provider, include_routing_query=True)
    nll = agent.conditioned_nll(prompt, target, memory)
    contrast_info = {}
    if contrast:
        wrong_memory = LoopMemory(memory.prefix_length, memory.slots,
                                  memory.loops, (swapped_tokens,))
        wrong_nll = agent.conditioned_nll(prompt, target, wrong_memory)
        contrast_info = {'contrast_loss': F.softplus(
            agent.config.train.bank_payload_contrast_margin + nll - wrong_nll),
            'swapped_source_nll': wrong_nll, 'swapped_ids': swapped_ids,
            'swapped_payload_bytes': sum(dim * 2 for dim in agent.config.memory.payload_dims)}
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
                    'supplied_positive': True} | contrast_info
