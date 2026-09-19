"""Counterfactual and candidate-free confirmation using only persisted compact codes."""
import argparse
import json
from pathlib import Path

import torch

from probe_stored_compaction import persist_codes
from sdkb.checkpoints import resolve_checkpoint
from sdkb.cluster_store import ClusterBank, state_fingerprint
from sdkb.compaction import SyntheticCompactor
from sdkb.data import load_episodes, counterfactual_multiuse, save_episodes
from sdkb.evaluation import build_shared_bank, score_answers
from sdkb.evaluation_adapter import load_frozen_agent
from sdkb.metrics import summarize_rows, counterfactual_metrics
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import config_from_run, autocast_context
from sdkb.trajectories import file_sha256


@torch.no_grad()
def run(source, fit, output):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        if not all((fit / f'{method}-stored.json').exists() for method in ('mean', 'trained')):
            raise ValueError('Wait for the completed frozen-compactor fit and initial stored evaluation')
        checkpoint = resolve_checkpoint(source, verify=True)
        state = torch.load(fit / 'resume.pt', weights_only=True, map_location='cpu')
        identity = state['identity']
        if (state['step'] != identity['steps'] or identity['checkpoint_manifest_sha256'] !=
                file_sha256(checkpoint / 'manifest.json')):
            raise ValueError('Compactor endpoint incomplete or source changed')
        config = config_from_run(checkpoint)
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        if state_fingerprint(agent.reader) != identity['reader_hash']:
            raise ValueError('Compactor reader identity changed')
        compactor = SyntheticCompactor(config.memory.payload_dims[0], identity['compactor']['width'], 1).to(agent.device).eval()
        compactor.load_state_dict(state['compactor'])
        episodes = load_episodes(fit / 'heldout/episodes.jsonl')
        groups = {'all': episodes, **{kind: [counterfactual_multiuse(e, kind) for e in episodes]
                                    for kind in ('permission', 'restoration')}}
        inputs = {'source_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                  'compactor_state_sha256': file_sha256(fit / 'resume.pt'),
                  'episodes_sha256': file_sha256(fit / 'heldout/episodes.jsonl'),
                  'script_sha256': file_sha256(__file__), 'max_new_tokens': 24, 'generation_worlds': 8}
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != inputs:
            raise ValueError('Confirmation inputs changed')
        atomic_json(output / 'inputs.json', inputs)
        stores, banks = {}, {}
        for kind, group in groups.items():
            path = fit / 'heldout/bank.sqlite' if kind == 'all' else output / f'{kind}.sqlite'
            stores[kind] = DiskStore(path)
            if kind != 'all':
                save_episodes(output / f'{kind}.jsonl', group)
                build_shared_bank(agent, stores[kind], group)
            banks[kind] = {}
            for method in ('mean', 'trained'):
                banks[kind][method] = (ClusterBank(stores[kind], view=method, reader_hash=identity['reader_hash'])
                                      if kind == 'all' else persist_codes(agent, compactor if method == 'trained' else None,
                                                                        stores[kind], group, method))
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Writer or compactor invoked during stored-only confirmation')
        agent.produce = forbidden
        compactor.forward = forbidden
        worlds = list(dict.fromkeys(e.environment for e in episodes))[:8]
        originals = {e.episode_id: e for e in episodes}
        for method in ('raw', 'mean', 'trained'):
            destination = output / f'{method}.json'
            if destination.exists():
                continue
            rows, generations = [], []
            with autocast_context(config):
                for kind, group in groups.items():
                    for index, episode in enumerate(group):
                        if stop_requested(output):
                            raise RuntimeError('Stored confirmation stopped; completed arms remain reusable')
                        prompt = agent.prompt_ids(episode.query)
                        conditions = (['all', 'none', 'zero_values'] +
                                      [f'drop_{i}' for i in range(len(episode.required_ids))]) if kind == 'all' else ['cf_' + kind]
                        for condition in conditions:
                            ids = episode.required_ids
                            if condition == 'none':
                                ids = ()
                            elif condition.startswith('drop_'):
                                ids = tuple(rid for i, rid in enumerate(ids) if i != int(condition[5:]))
                            memory, accounting = None, []
                            if ids:
                                session = read_session(agent, stores[kind], prompt, namespace='global',
                                    generation='frozen-v1', query_time=episode.query_time, oracle_ids=ids,
                                    cluster_bank=banks[kind].get(method), ablate_values=condition == 'zero_values')
                                memory, accounting = session.memory, session.payload_accounting
                            row = {'episode': episode.episode_id, 'environment': episode.environment,
                                   'task_family': episode.task_family, 'condition': condition, 'answer': episode.answer,
                                   'selected_ids': list(ids), 'payload_accounting': accounting,
                                   'complete_support': any(set(g) <= set(ids) for g in
                                                           (episode.sufficient_groups or (episode.required_ids,))),
                                   **score_answers(agent, prompt, memory, episode.answer, episode.choices)}
                            if kind != 'all':
                                row['counterfactual_should_change'] = episode.answer != originals[episode.episode_id].answer
                            rows.append(row)
                            if kind == 'all' and episode.environment in worlds and condition in {'all', 'none', 'zero_values'}:
                                prediction = agent.generate_from_memory(prompt, memory, max_new_tokens=24)
                                generations.append({k: row[k] for k in ('episode', 'environment', 'task_family', 'condition', 'answer')} |
                                                   {'prediction': prediction, 'exact_match': prediction == episode.answer})
                        if (index + 1) % 64 == 0:
                            print(json.dumps({'method': method, 'variant': kind, 'queries': index + 1}), flush=True)
            families = sorted({e.task_family for e in episodes})
            atomic_json(destination, {'inputs': inputs, 'method': method, 'rows': rows, 'generation_rows': generations,
                'by_family': {f: summarize_rows([r for r in rows if r['task_family'] == f]) for f in families},
                'counterfactuals_by_family': {f: counterfactual_metrics([r for r in rows if r['task_family'] == f],
                                                                        ('cf_permission', 'cf_restoration')) for f in families},
                'generation_by_family': {f: {c: {'n': len(selected), 'exact_match': sum(r['exact_match'] for r in selected) / len(selected)}
                     for c in ('all', 'none', 'zero_values')
                     if (selected := [r for r in generations if r['task_family'] == f and r['condition'] == c])} for f in families},
                'code_storage': {kind: banks[kind][method].sizes() for kind in groups} if method != 'raw' else None,
                'notice': 'Oracle selected clusters; exact raw subset fallback. No learned routing, net disk savings or agent claim.'})
            print(json.dumps({'completed': method}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.fit, args.output)
