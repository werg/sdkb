"""Greedy generation from an existing frozen bank, without candidate answers.

This is a synthetic diagnostic, not an agent execution benchmark. The checkpoint,
episodes and bank must be the matching artifacts from a stored-transfer evaluation.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json

import torch
from safetensors.torch import load_model

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes, evidence_ids
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import autocast_context, config_from_run
from sdkb.trajectories import file_sha256


@torch.no_grad()
def evaluate(run, bank, episodes_path, worlds, max_new_tokens):
    if worlds < 1 or max_new_tokens < 1:
        raise ValueError('World and generation budgets must be positive')
    checkpoint = resolve_checkpoint(run, verify=True)
    config = config_from_run(run)
    if config.train.arm != 'memory' or config.train.retrieval != 'oracle':
        raise ValueError('This diagnostic requires the controlled oracle-memory arm')
    if not bank.is_file():
        raise FileNotFoundError('Supply an existing frozen bank')
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    agent = SDKBAgent(config).to(config.train.device).eval()
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)

    def forbidden_writer(*args, **kwargs):
        raise AssertionError('Generation must consume stored payloads without calling the writer')
    agent.produce = forbidden_writer
    store = DiskStore(bank)
    episodes = load_episodes(episodes_path)
    selected_worlds = list(dict.fromkeys(e.environment for e in episodes))[:worlds]
    rows = []
    with autocast_context(config):
        for episode in episodes:
            if episode.environment not in selected_worlds:
                continue
            prompt = agent.prompt_ids(episode.query)
            plans = None
            for condition in ('all', 'zero_values', 'none'):
                memory = None
                if condition != 'none':
                    session = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                           query_time=episode.query_time,
                                           oracle_ids=tuple(evidence_ids(episode, config.train.evidence_scope)),
                                           ablate_values=condition == 'zero_values', fixed_plans=plans)
                    memory = session.memory
                    if condition == 'all':
                        plans = session.plans
                prediction = agent.generate_from_memory(prompt, memory, max_new_tokens=max_new_tokens)
                rows.append(dict(episode=episode.episode_id, environment=episode.environment,
                                 task_family=episode.task_family, condition=condition,
                                 answer=episode.answer, prediction=prediction,
                                 exact_match=prediction == episode.answer))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task_family'], row['condition'])].append(row['exact_match'])
    families = sorted({r['task_family'] for r in rows})
    return {
        'protocol': 'Greedy stored-only generation; no candidate answers supplied; writer disabled.',
        'notice': 'First requested validation worlds, chosen by file order. Exact strings after '
                  'decoder whitespace stripping; no agent execution or capacity-substitution claim.',
        'max_new_tokens': max_new_tokens,
        'worlds': len(selected_worlds),
        'inputs': {'checkpoint': str(checkpoint),
                   'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                   'bank': str(bank), 'bank_sha256': file_sha256(bank),
                   'episodes': str(episodes_path), 'episodes_sha256': file_sha256(episodes_path)},
        'by_family': {f: {c: {'n': len(groups[(f, c)]),
                              'exact_match': sum(groups[(f, c)]) / len(groups[(f, c)])}
                          for c in ('all', 'zero_values', 'none')} for f in families},
        'rows': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=24)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args.run, args.bank, args.episodes, args.worlds, args.max_new_tokens)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
