"""Counterfactual action sensitivity under captured full-bank learned read plans."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes, counterfactual_multiuse, save_episodes
from sdkb.evaluation import build_shared_bank
from sdkb.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from sdkb.frozen_scoring import FrozenScorer
from sdkb.offline_bank import canonical_json
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.sessions import read_session
from sdkb.store import DiskStore, ReadPlan, Selection
from sdkb.training import config_from_run, autocast_context
from sdkb.trajectories import file_sha256


@torch.no_grad()
def run(source, reference_path, episodes_file, router, count_policy, output, generations_file=None):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        reference = json.loads(reference_path.read_text())
        if not reference['global_bank']:
            raise ValueError('Supply the completed full-bank reference')
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        config.memory.neighbors = [2]
        torch.set_num_threads(config.train.threads)
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=router, independent_routing_query=True)
        count = attach_read_count_policy(agent, checkpoint, count_policy)
        agent.requires_grad_(False)
        prior = reference['inputs']
        if (prior['source_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json') or
                prior['episodes_sha256'] != file_sha256(episodes_file) or prior['routing_probe'] != adapter or
                prior['read_count_policy'] != count):
            raise ValueError('Counterfactual reference identity differs')
        identity = {'reference_sha256': file_sha256(reference_path), 'source_inputs': prior,
                    'script_sha256': file_sha256(__file__),
                    'selection': 'Captured original full-bank learned selections; no counterfactual reranking'}
        generation_reference = reference
        if generations_file is not None:
            generation_reference = json.loads(generations_file.read_text())
            if (generation_reference['source_inputs'] != prior or
                    generation_reference['source_reference_sha256'] != file_sha256(reference_path)):
                raise ValueError('Expanded generation reference differs')
            identity['generation_reference_sha256'] = file_sha256(generations_file)
            identity['generation_worlds'] = generation_reference['worlds']
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Counterfactual inputs changed')
        atomic_json(output / 'inputs.json', identity)
        original = load_episodes(episodes_file)
        baseline = {r['episode']: r for r in reference['rows'] if r['condition'] == 'all'}
        generated = {r['episode']: r for r in generation_reference['generation_rows'] if r['condition'] == 'all'}
        if set(baseline) != {e.episode_id for e in original}:
            raise ValueError('Counterfactual episode grid differs')
        groups, stores = {}, {}
        for kind in ('permission', 'restoration'):
            groups[kind] = [counterfactual_multiuse(e, kind) for e in original]
            data_path = output / f'{kind}.jsonl'
            save_episodes(data_path, groups[kind])
            stores[kind] = DiskStore(output / f'{kind}.sqlite')
            writer_identity = hashlib.sha256(canonical_json({'source': prior['source_manifest_sha256'],
                'adapter': adapter['probe_sha256'], 'episodes': file_sha256(data_path)}).encode()).hexdigest()
            build_shared_bank(agent, stores[kind], groups[kind], writer_identity=writer_identity)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Counterfactual inference invoked a writer or compactor')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        scorer = FrozenScorer(agent)
        for kind, group in groups.items():
            destination = output / f'{kind}.json'
            if destination.exists():
                if json.loads(destination.read_text())['inputs'] != identity:
                    raise ValueError('Completed counterfactual identity differs')
                continue
            rows, generations = [], []
            with autocast_context(config):
                for e, old in zip(group, original, strict=True):
                    if e.task_family != 'multiuse/action':
                        continue
                    if stop_requested(output):
                        raise RuntimeError('Counterfactual evaluation stopped; completed variants remain reusable')
                    if e.query != old.query or e.required_ids != old.required_ids or e.query_time != old.query_time:
                        raise ValueError('Counterfactual changed the causal query or evidence identities')
                    previous = baseline[e.episode_id]
                    if previous['answer'] != old.answer:
                        raise ValueError('Reference answer differs')
                    plan = ReadPlan('global', 's0', 'frozen-v1', 'research', e.query_time,
                                    tuple(Selection(rid, 0.) for rid in previous['selected_ids']))
                    prompt = agent.prompt_ids(e.query)
                    session = read_session(agent, stores[kind], prompt, namespace='global', generation='frozen-v1',
                                           query_time=e.query_time, fixed_plans=[[plan]])
                    selected = list(dict.fromkeys(r for ids in session.selected_ids for r in ids))
                    if selected != previous['selected_ids']:
                        raise ValueError('Captured counterfactual selection changed')
                    score = scorer.score(prompt, session.memory, e.answer, e.choices)
                    rows.append({'episode': e.episode_id, 'environment': e.environment, 'variant': kind,
                        'answer': e.answer, 'selected_ids': selected, **score,
                        'should_change': old.answer != e.answer,
                        'both_correct': previous['choice_correct'] and score['choice_correct'],
                        'prediction_changed': previous['predicted_action'] != score['predicted_action']})
                    if e.episode_id in generated:
                        prediction = scorer.generate(prompt, session.memory, max_new_tokens=prior['max_new_tokens'])
                        old_generation = generated[e.episode_id]
                        generations.append({'episode': e.episode_id, 'environment': e.environment, 'variant': kind,
                            'answer': e.answer, 'prediction': prediction, 'exact_match': prediction == e.answer,
                            'should_change': old.answer != e.answer,
                            'both_correct': old_generation['exact_match'] and prediction == e.answer,
                            'prediction_changed': old_generation['prediction'] != prediction})
            def summary(group, metric):
                changed = [r for r in group if r['should_change']]
                unchanged = [r for r in group if not r['should_change']]
                return {'n': len(group), 'correct': sum(r[metric] for r in group),
                    'changed_pairs': len(changed), 'both_correct_on_change': sum(r['both_correct'] for r in changed),
                    'unchanged_pairs': len(unchanged), 'false_changes': sum(r['prediction_changed'] for r in unchanged)}
            atomic_json(destination, {'inputs': identity, 'bank_sha256': file_sha256(stores[kind].path),
                'choices': summary(rows, 'choice_correct'), 'generation': summary(generations, 'exact_match'),
                'rows': rows, 'generation_rows': generations, 'decoder_reuse': scorer.report(),
                'notice': 'Sensitivity to stored rule values under fixed learned retrieval, not end-to-end rerouting or agent success.'})
            print(canonical_json({'completed_variant': kind, 'choices': summary(rows, 'choice_correct'),
                                  'generation': summary(generations, 'exact_match')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'reference', 'episodes', 'router', 'count-policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--generations', type=Path)
    args = parser.parse_args()
    run(args.source, args.reference, args.episodes, args.router, args.count_policy, args.output, args.generations)
