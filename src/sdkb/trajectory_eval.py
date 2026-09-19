"""Write-once, serialize/reload, stored-only teacher-continuation evaluation.

Likelihood is imitation/conditioning, not task execution success. There is no live
teacher or tool execution, and the reader never calls the source writer.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import math
import random
import time

import torch
from safetensors.torch import load_model
from .agent import SDKBAgent
from .checkpoints import resolve_checkpoint
from .episode_index import EpisodeIndex
from .sessions import read_session
from .store import DiskStore
from .training import autocast_context, config_from_run, persist_outputs, stored_channel, resource_report


@torch.no_grad()
def build_teacher_bank(agent, store, episodes):
    if agent.training:
        raise ValueError('Freeze writer in eval mode before materializing memory')
    sources = {}
    for e in episodes:
        for s in e.supports:
            if s.record_id in sources and sources[s.record_id] != s:
                raise ValueError('Conflicting source content')
            sources[s.record_id] = s
    outputs = {}
    if agent.config.train.arm in {'memory', 'direct_latent'}:
        with autocast_context(agent.config):
            for rid, s in sources.items():
                outputs[rid] = tuple(t.detach().cpu() for t in stored_channel(agent, agent.produce(agent.text_ids(s.text, source=True))))
        ids, seen = sorted(outputs), set()
        peers = {rid: ids[(i + 1) % len(ids)] for i, rid in enumerate(ids)}
        for e in episodes:
            for s in e.supports:
                key = (e.environment, s.record_id)
                if key in seen:
                    continue
                seen.add(key)
                original, peer = outputs[s.record_id], outputs[peers[s.record_id]]
                wrong = tuple(original[i] if i % 2 == 0 else peer[i] for i in range(len(original)))
                for variant, tensors in [('all', original), ('wrong_values', wrong)]:
                    persist_outputs(store, agent, s, tensors, variant + '/' + e.environment, 'teacher-eval-v1')
    return dict(unique_sources=len(sources), writer_calls=len(outputs),
                wrong_values_distinct_record_available=len(sources) > 1,
                wrong_values_policy='Cyclic distinct-record payload permutation; semantic disagreement not guaranteed.')


def paired_nll_benefit(rows, draws=500):
    pairs, groups = defaultdict(dict), defaultdict(list)
    for r in rows:
        pairs[r['episode']][r['condition']] = r
    for p in pairs.values():
        if 'all' in p and 'none' in p:
            groups[p['all']['trajectory']].append(p['none']['mean_nll'] - p['all']['mean_nll'])
    if not groups:
        return {'gain': None, 'ci95': None}
    values, rng, samples = list(groups.values()), random.Random(0), []
    for _ in range(draws):
        selected = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
    samples.sort()
    return dict(gain=sum(map(sum, values)) / sum(map(len, values)),
                ci95=[samples[int(draws * .025)], samples[min(draws - 1, int(draws * .975))]],
                trajectories=len(groups), sign='positive = reduced mean-token NLL with memory',
                scope='Bootstrap trajectories, not training seeds.')


@torch.no_grad()
def stored_teacher_evaluation(agent, store, episodes, *, generate_tokens=0):
    if agent.training or not episodes:
        raise ValueError('Nonempty evaluation set and eval-mode agent required')
    rows = []
    with autocast_context(agent.config):
        for e in episodes:
            for condition in ('all', 'none', 'zero_values', 'wrong_values'):
                arm = agent.config.train.arm
                evidence = '\n'.join(s.text for s in e.supports if s.record_id in e.required_ids)
                prompt = agent.prompt_ids(e.query, evidence if arm == 'oracle_text' and condition != 'none' else '')
                memory, selected = None, []
                if arm in {'memory', 'direct_latent'} and condition != 'none':
                    namespace = ('wrong_values' if condition == 'wrong_values' else 'all') + '/' + e.environment
                    session = read_session(agent, store, prompt, namespace=namespace, generation='teacher-eval-v1',
                                           query_time=e.query_time,
                                           oracle_ids=e.required_ids if agent.config.train.retrieval == 'oracle' else None,
                                           ablate_values=condition == 'zero_values')
                    memory, selected = session.memory, session.selected_ids
                elif arm == 'shared_compute':
                    memory = agent.shared_compute_tokens(prompt)
                target = agent.target_ids(e.answer)
                nll = float(agent.conditioned_nll(prompt, target, memory, reduction='sum'))
                r = dict(episode=e.episode_id, trajectory=e.environment, dataset=e.provenance.get('dataset'),
                         condition=condition, token_count=target.numel(), sequence_nll=nll, mean_nll=nll / target.numel(),
                         selected_ids=selected, support_annotation=e.support_annotation)
                if generate_tokens:
                    prediction = agent.generate_from_memory(prompt, memory, max_new_tokens=generate_tokens)
                    r.update(prediction=prediction, reference_exact_match=prediction.strip() == e.answer.strip())
                rows.append(r)
    summary = {}
    for c in ('all', 'none', 'zero_values', 'wrong_values'):
        subset = [r for r in rows if r['condition'] == c]
        nll = sum(r['sequence_nll'] for r in subset) / sum(r['token_count'] for r in subset)
        summary[c] = dict(episodes=len(subset), token_weighted_nll=nll,
                          perplexity=math.exp(nll) if nll < 50 else None,
                          macro_mean_nll=sum(r['mean_nll'] for r in subset) / len(subset))
    return dict(protocol='Write-once/serialize/reload, time-filtered per-trajectory stored-only reads.',
                summary=summary, paired_memory_nll_benefit=paired_nll_benefit(rows), rows=rows,
                notice='Teacher likelihood is not task success. Payload interventions affect latent arms only; '
                       'supplied context is not annotated sufficient. No environment commands are executed.')


def evaluate_teacher_run(run, episodes_file, *, max_episodes=64, generate_tokens=0, output=None):
    if max_episodes < 1 or generate_tokens < 0:
        raise ValueError('Invalid evaluation limits')
    config = config_from_run(Path(run))
    torch.set_num_threads(config.train.threads)
    agent = SDKBAgent(config).to(config.train.device).eval()
    load_model(agent, str(resolve_checkpoint(run, verify=True) / 'model.safetensors'), device=config.train.device)
    index = EpisodeIndex(episodes_file)
    ids = random.Random(config.train.seed + 101).sample(range(len(index)), min(max_episodes, len(index)))
    episodes = [index[i] for i in ids]
    directory = Path(output) if output else Path(run) / ('teacher-eval-' + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    store = DiskStore(directory / 'bank.sqlite')
    writes = build_teacher_bank(agent, store, episodes)
    result = stored_teacher_evaluation(agent, DiskStore(store.path), episodes, generate_tokens=generate_tokens)
    result.update(write_phase=writes, resources=resource_report(), store=store.sizes())
    (directory / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return {k: v for k, v in result.items() if k != 'rows'} | {'directory': str(directory)}
