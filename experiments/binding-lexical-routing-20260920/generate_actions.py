"""Frozen stored-latent action generation under externally indexed lexical selection."""
import argparse
import json
from pathlib import Path
import runpy

import torch

from sdkb.archiving import ensure_free
from sdkb.checkpoints import _atomic_text, resolve_checkpoint, stop_on_signal
from sdkb.data import load_episodes
from sdkb.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from sdkb.frozen_scoring import FrozenScorer
from sdkb.operations import run_lock, stop_requested
from sdkb.sessions import read_session
from sdkb.store import DiskStore, ReadPlan, Selection
from sdkb.training import config_from_run, autocast_context
from sdkb.trajectories import file_sha256


@torch.no_grad()
def run(source, corpus, bank, reference, lexical, index, router, count_policy, output):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        if stop_requested(output):
            raise RuntimeError('Lexical generation stop requested')
        checkpoint = resolve_checkpoint(source, verify=True)
        prior = json.loads(reference.read_text())
        ranking = json.loads(lexical.read_text())
        helper = Path(__file__).parents[2]/'scripts/evaluate_lexical_routing.py'
        identity = {'source_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                    'episodes_sha256': file_sha256(corpus), 'bank_sha256': file_sha256(bank),
                    'reference_sha256': file_sha256(reference), 'lexical_sha256': file_sha256(lexical),
                    'index_sha256': file_sha256(index), 'script_sha256': file_sha256(__file__),
                    'ranking_script_sha256': file_sha256(helper), 'max_new_tokens': 24}
        for key in ('source_manifest_sha256', 'episodes_sha256', 'bank_sha256'):
            if identity[key] != prior['inputs'][key]:
                raise ValueError('Generation source/corpus/bank differs from reference')
        for key in ('bank_sha256', 'reference_sha256', 'index_sha256'):
            if identity[key] != ranking['inputs'][key]:
                raise ValueError('Lexical ranking identity differs')
        if identity['ranking_script_sha256'] != ranking['inputs']['script_sha256']:
            raise ValueError('Lexical ranking implementation changed')
        config = config_from_run(checkpoint)
        config.memory.neighbors = [2]
        torch.set_num_threads(config.train.threads)
        agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=router, independent_routing_query=True)
        count = attach_read_count_policy(agent, checkpoint, count_policy)
        if adapter != prior['inputs']['routing_probe'] or count != prior['inputs']['read_count_policy']:
            raise ValueError('Frozen address/count overlay differs')
        identity.update(routing_probe=adapter, read_count_policy=count)
        agent.requires_grad_(False)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Lexical generation invoked a source writer or compactor')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        def save(path, value):
            ensure_free(output, config.train.min_free_disk_bytes)
            _atomic_text(path, json.dumps(value, indent=2)+'\n')
        plans = {r['episode']: r for r in ranking['rows'] if r['top_k'] == 2 and r['task_family'] == 'multiuse/action'}
        episodes = [e for e in load_episodes(corpus) if e.episode_id in plans]
        if len(episodes) != len(plans):
            raise ValueError('Lexical action query set differs')
        conditions = ('all', 'selected_pair', 'none', 'zero_values')
        expected = [(e, c) for e in episodes for c in conditions]
        progress, rows = output/'progress.json', []
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved['inputs'] != identity:
                raise ValueError('Generation progress identity changed')
            rows = saved['rows']
        if len(rows) > len(expected):
            raise ValueError('Extra generation progress')
        for row, (e, condition) in zip(rows, expected):
            if (row['episode'], row['condition'], row['answer']) != (e.episode_id, condition, e.answer):
                raise ValueError('Generation progress prefix differs')
            if row['exact_match'] != (row['prediction'].strip() == e.answer.strip()):
                raise ValueError('Saved exact match differs')
        old = {(r['episode'], r['condition']): r for r in prior['generation_rows']}
        for row in rows:
            if row['condition'] in ('selected_pair', 'none') and row['prediction'] != old[row['episode'], row['condition']]['prediction']:
                raise ValueError('Saved control prediction differs')
        records = json.loads(index.read_text())['records']
        rank = runpy.run_path(str(helper))['rank']
        scorer, store = FrozenScorer(agent), DiskStore(bank)
        with autocast_context(config):
            for e, condition in expected[len(rows):]:
                if stop['signal'] is not None or stop_requested(output):
                    save(progress, {'inputs': identity, 'rows': rows})
                    raise RuntimeError('Lexical generation stopped; completed strings are resumable')
                selected = [r['record_id'] for r in rank(records, e.query, query_time=e.query_time, top_k=2)]
                if selected != plans[e.episode_id]['selected_ids']:
                    raise ValueError('Stored index ranking differs from captured selection')
                prompt = agent.prompt_ids(e.query)
                memory = None
                if condition != 'none':
                    ids = e.required_ids if condition == 'selected_pair' else selected
                    plan = ReadPlan('global', 's0', 'frozen-v1', 'research', e.query_time,
                                    tuple(Selection(record_id, 0.) for record_id in ids))
                    memory = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                          query_time=e.query_time, fixed_plans=[[plan]],
                                          ablate_values=condition == 'zero_values').memory
                prediction = scorer.generate(prompt, memory, max_new_tokens=24)
                if condition in ('selected_pair', 'none') and prediction != old[e.episode_id, condition]['prediction']:
                    raise ValueError('Frozen oracle/no-memory control prediction differs')
                rows.append({'episode': e.episode_id, 'environment': e.environment, 'task_family': e.task_family,
                             'condition': condition, 'answer': e.answer, 'prediction': prediction,
                             'exact_match': prediction.strip() == e.answer.strip()})
                if len(rows) % 32 == 0:
                    save(progress, {'inputs': identity, 'rows': rows})
                    print(json.dumps({'completed': len(rows), 'total': len(expected)}), flush=True)
        if file_sha256(bank) != identity['bank_sha256']:
            raise ValueError('Frozen bank changed')
        summary = {c: {'n': len(group := [r for r in rows if r['condition'] == c]),
                       'correct': sum(r['exact_match'] for r in group)} for c in conditions}
        save(output/'results.json', {'inputs': identity, 'generation_rows': rows, 'summary': summary,
             'notice': 'Known-corpus action-only comparator. Query-text lexical features select two records; '
                       'read-plan scores are fixed to zero, as in the selected-ID generation diagnostic. '
                       'Reader/decoder/payloads stay frozen. Extra lexical index bytes are not matched to learned keys.'})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'corpus', 'bank', 'reference', 'lexical', 'index', 'router', 'count-policy', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.corpus, args.bank, args.reference, args.lexical, args.index,
        args.router, args.count_policy, args.output)
