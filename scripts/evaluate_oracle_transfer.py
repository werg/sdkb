"""Resumable offline banks followed by frozen oracle transfer and free generation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import re

import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes, counterfactual_multiuse, evidence_ids
from sdkb.evaluation import build_shared_bank, build_persistent_codes, stored_transfer_evaluation
from sdkb.evaluation_adapter import load_frozen_agent
from sdkb.frozen_scoring import FrozenScorer
from sdkb.metrics import counterfactual_metrics
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import config_from_run, autocast_context
from sdkb.trajectories import file_sha256


def identifier_counterfactual(episode):
    """Synthetic endpoint-only intervention, preserving IDs, rules and causal time."""
    if not episode.task_family.startswith('multiuse/'):
        raise ValueError('Generated multiuse episode required')
    def invert(value):
        if re.fullmatch(r'api_[0-9a-f]{6}', value) is None:
            raise ValueError('Expected generated six-hex endpoint')
        return 'api_'+''.join(format(int(c, 16) ^ 15, 'x') for c in value[4:])
    sources = []
    for source in episode.supports:
        text = source.text
        if source.kind == 'permission':
            text, count = re.subn(r'(?<=Its endpoint is )api_[0-9a-f]{6}(?=\.)',
                                 lambda match: invert(match[0]), text)
            if count != 1:
                raise ValueError('Malformed generated endpoint source')
        sources.append(replace(source, text=text))
    identifier = episode.task_family == 'multiuse/identifier'
    return replace(episode, supports=tuple(sources),
                   answer=invert(episode.answer) if identifier else episode.answer,
                   choices=tuple(map(invert, episode.choices)) if identifier else episode.choices)


@torch.no_grad()
def run(source, episodes_path, output, *, compact_method='raw'):
    if compact_method not in {'raw', 'mean', 'trained'}:
        raise ValueError('Choose raw, mean or trained code reads')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                    'episodes_sha256': file_sha256(episodes_path), 'script_sha256': file_sha256(__file__),
                    'max_new_tokens': 24, 'routing': 'oracle required sources', 'compact_method': compact_method}
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != identity:
            raise ValueError('Oracle evaluation identity changed')
        atomic_json(output/'inputs.json', identity)
        if (output/'results.json').exists():
            completed = json.loads((output/'results.json').read_text())
            if completed['inputs'] != identity:
                raise ValueError('Completed oracle evaluation identity changed')
            if not (output/'bank.sqlite').is_file() or file_sha256(output/'bank.sqlite') != completed['bank_sha256']:
                raise ValueError('Completed oracle evaluation bank changed')
            return
        config = config_from_run(checkpoint)
        if config.train.arm != 'memory' or config.train.retrieval != 'oracle' or config.train.evidence_scope != 'required':
            raise ValueError('This diagnostic requires oracle required-source memory')
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        if compact_method != 'raw':
            if len(config.memory.payload_dims) != 1:
                raise ValueError('Compact confirmation requires one space')
            if compact_method == 'trained' and agent.compactor is None:
                raise ValueError('Trained-code confirmation requires a checkpointed compactor')
            config.memory.compaction = 'mean' if compact_method == 'mean' else 'synthetic'
        episodes = load_episodes(episodes_path)
        variants = {k: [counterfactual_multiuse(e, k) for e in episodes] for k in ('permission', 'restoration')}
        variants['identifier'] = [identifier_counterfactual(e) for e in episodes]
        store = DiskStore(output/'bank.sqlite')
        writes, banks, code_storage = {}, {}, {}
        for name, group in [('global', episodes), *variants.items()]:
            writes[name] = build_shared_bank(agent, store, group, namespace=name,
                writer_identity=identity['checkpoint_manifest_sha256'])
            if compact_method != 'raw':
                banks[name], code_storage[name] = build_persistent_codes(agent, store, group, namespace=name,
                                                                             view_name='oracle-'+name+'-'+compact_method)
            if stop_requested(output):
                raise RuntimeError('Stopped after atomic bank publication')
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Oracle inference called writer or compactor')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        store = DiskStore(store.path)
        plans = {}
        def progress(event, *, phase='scoring', variant='global'):
            if stop_requested(output):
                raise RuntimeError('Stopped during reproducible stored evaluation')
            print(json.dumps({'phase': phase, 'variant': variant, **event}), flush=True)
        scores = stored_transfer_evaluation(agent, store, episodes, drop_supports=True,
                                            capture_plans=plans, progress=progress, cluster_bank=banks.get('global'),
                                            use_codes_for_all_conditions=compact_method != 'raw')
        originals = {e.episode_id: e for e in episodes}
        for name, group in variants.items():
            changed = stored_transfer_evaluation(agent, store, group, namespace=name,
                full_evidence_only=True, fixed_plans_by_episode=plans,
                progress=lambda event, variant=name: progress(event, variant=variant), cluster_bank=banks.get(name),
                use_codes_for_all_conditions=compact_method != 'raw')
            for row in changed['rows']:
                row['condition'] = 'cf_'+name
                row['counterfactual_should_change'] = row['answer'] != originals[row['episode']].answer
            scores['rows'].extend(changed['rows'])
        scorer, rows = FrozenScorer(agent), []
        with autocast_context(config):
            for name, group in [('global', episodes), *variants.items()]:
                for index, e in enumerate(group, 1):
                    if stop_requested(output):
                        raise RuntimeError('Stopped during reproducible free generation')
                    prompt = agent.prompt_ids(e.query)
                    for condition in (('all', 'zero_values', 'none') if name == 'global' else ('cf_'+name,)):
                        memory, accounting = None, []
                        if condition != 'none':
                            fixed = [[replace(p, namespace=name) for p in step] for step in plans[e.episode_id]]
                            session = read_session(agent, store, prompt, namespace=name, generation='frozen-v1',
                                query_time=e.query_time, oracle_ids=tuple(evidence_ids(e, 'required')),
                                ablate_values=condition == 'zero_values', fixed_plans=fixed, cluster_bank=banks.get(name))
                            memory, accounting = session.memory, session.payload_accounting
                        prediction = scorer.generate(prompt, memory, max_new_tokens=24)
                        row = {'episode': e.episode_id, 'environment': e.environment, 'task_family': e.task_family,
                               'condition': condition, 'answer': e.answer, 'prediction': prediction,
                               'exact_match': prediction == e.answer, 'payload_accounting': accounting}
                        if name != 'global':
                            row['counterfactual_should_change'] = e.answer != originals[e.episode_id].answer
                        rows.append(row)
                    if index % 32 == 0 or index == len(group):
                        progress({'completed_queries': index, 'total_queries': len(group),
                                  'completed_generations': len(rows), 'total_generations': 6*len(episodes)},
                                 phase='generation', variant=name)
        families = sorted({e.task_family for e in episodes})
        summary = {f: {c: {'n': len(group), 'correct': sum(r['exact_match'] for r in group)}
                      for c in sorted({r['condition'] for r in rows})
                      if (group := [r for r in rows if r['task_family'] == f and r['condition'] == c])}
                   for f in families}
        cf_names = tuple('cf_'+name for name in variants)
        scores['counterfactuals_by_family'] = {f: counterfactual_metrics(
            [r for r in scores['rows'] if r['task_family'] == f], cf_names) for f in families}
        generation_cf = {f: counterfactual_metrics(
            [dict(r, choice_correct=r['exact_match'], predicted_action=r['prediction'])
             for r in rows if r['task_family'] == f], cf_names) for f in families}
        atomic_json(output/'results.json', {'inputs': identity, 'writes': writes, 'scores': scores,
            'generation_rows': rows, 'generation_summary': summary, 'generation_counterfactuals': generation_cf,
            'bank_sha256': file_sha256(store.path), 'code_storage': code_storage,
            'notice': 'Oracle-selected causal sources, frozen stored-only reads, all-world free generation. '
                      'Rule-type changes use fixed original plans. No learned-routing or agent-success claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--compact-method', choices=['raw', 'mean', 'trained'], default='raw')
    args = parser.parse_args()
    run(args.source, args.episodes, args.output, compact_method=args.compact_method)
