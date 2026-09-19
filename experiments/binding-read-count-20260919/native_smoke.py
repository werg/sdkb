"""Exercise the frozen count policy with native LFM stored reads and no writer access."""
import argparse
from pathlib import Path

import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import make_multiuse_world, save_episodes
from sdkb.evaluation import build_shared_bank
from sdkb.evaluation_adapter import load_frozen_agent, attach_read_count_policy
from sdkb.operations import atomic_json
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import config_from_run, autocast_context, environment_report
from sdkb.trajectories import file_sha256


def smoke(source, router, policy, output):
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(checkpoint)
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    agent, adapter = load_frozen_agent(config, checkpoint, routing_probe=router, independent_routing_query=True)
    episodes = [e for i in range(2) for e in make_multiuse_world(i, split='read-count-native-smoke-20260919', bindings=2)]
    output.mkdir(parents=True, exist_ok=False)
    save_episodes(output / 'episodes.jsonl', episodes)
    store = DiskStore(output / 'bank.sqlite')
    with torch.no_grad(), autocast_context(config):
        source_ids = agent.text_ids(episodes[0].supports[0].text, source=True)
        before = agent.produce(source_ids)
        count_policy = attach_read_count_policy(agent, checkpoint, policy)
        after = agent.produce(source_ids)
        if not all(torch.equal(a, b) for a, b in zip(before, after, strict=True)):
            raise ValueError('Count policy changed offline writer output')
        build_shared_bank(agent, store, episodes)
        def forbidden(*args, **kwargs):
            raise AssertionError('Stored read called the writer')
        agent.produce = forbidden
        universe = frozenset(s.record_id for e in episodes for s in e.supports)
        rows = []
        for e in episodes:
            allowed = frozenset(s.record_id for s in e.supports)
            session = read_session(agent, store, agent.prompt_ids(e.query), namespace='global',
                                   generation='frozen-v1', query_time=e.query_time, exclude_ids=universe - allowed)
            selected = session.selected_ids[0]
            if not set(selected) <= allowed or not 1 <= len(selected) <= 2:
                raise ValueError('Read count or eligibility invalid')
            rows.append({'episode': e.episode_id, 'family': e.task_family, 'requested': len(selected),
                         'required_count': len(e.required_ids), 'selected': selected})
    atomic_json(output / 'results.json', {'environment': environment_report(), 'adapter': adapter,
                'policy': count_policy, 'episodes_sha256': file_sha256(output / 'episodes.jsonl'),
                'queries': len(rows), 'correct_counts': sum(r['requested'] == r['required_count'] for r in rows),
                'offline_writer_outputs_identical': True, 'stored_writer_calls': 0, 'rows': rows,
                'notice': 'Native mechanics/count smoke only; no downstream capability claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'router', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    smoke(args.source, args.router, args.policy, args.output)
