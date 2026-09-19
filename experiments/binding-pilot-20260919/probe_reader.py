"""Locate bit signal after the reader using frozen stored counterfactual banks."""
from dataclasses import replace
from pathlib import Path
import argparse
import json

import torch
from safetensors.torch import load_model

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import config_from_run, autocast_context
from probe_payloads import linear_probe


@torch.no_grad()
def probe(run, episodes_file, directory):
    config = config_from_run(run)
    if config.memory.read_timing != 'loop_boundary' or config.memory.read_steps != 1:
        raise ValueError('This probe expects one native inter-pass read')
    torch.set_num_threads(config.train.threads)
    agent = SDKBAgent(config).to(config.train.device).eval()
    checkpoint = resolve_checkpoint(run, verify=True)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reader probe must use existing stored payloads')
    agent.produce = forbidden
    episodes = load_episodes(episodes_file)
    store = DiskStore(directory / 'bank.sqlite')
    groups = {}
    with autocast_context(config):
        for e in episodes:
            family = e.task_family.split('/')[1]
            if family == 'identifier':
                continue
            prompt = agent.prompt_ids(e.query)
            original = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                    query_time=e.query_time, oracle_ids=e.required_ids)
            a = original.memory.events[0].flatten().float().cpu()
            for kind in (('permission', 'restoration') if family == 'action' else (family,)):
                namespace = 'flip-' + kind
                plans = [[replace(p, namespace=namespace) for p in step] for step in original.plans]
                variant = read_session(agent, store, prompt, namespace=namespace, generation='frozen-v1',
                    query_time=e.query_time, oracle_ids=e.required_ids, fixed_plans=plans)
                b = variant.memory.events[0].flatten().float().cpu()
                bit = e.allowed_capability if kind == 'permission' else int(e.restore)
                pairs, worlds = groups.setdefault(f'{family}_query/{kind}_bit', ([], []))
                pairs.append(torch.stack((a, b) if bit == 0 else (b, a)))
                worlds.append(e.environment)
    return {'checkpoint': str(checkpoint), 'evaluation': str(directory),
            'probes': {name: linear_probe(torch.stack(pairs), worlds) for name, (pairs, worlds) in groups.items()},
            'notice': 'Diagnostic readout of returned soft tokens, not deployed answer accuracy. '
                      'No writer calls; fixed query/IDs/read plans; world-disjoint probe fit and evaluation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(args.run, args.episodes, args.evaluation), indent=2) + '\n')
