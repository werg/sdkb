"""Diagnostic linear readout of stored bit information; no model or writer calls.

Counterfactual pairs share IDs/context and differ in one rule bit. Split by world
before fitting; neither member of a held-out pair enters the probe fit. A failed
linear probe does not establish absence of nonlinear information.
"""
from pathlib import Path
import argparse
import json
import random
import re

import torch

from sdkb.data import load_episodes
from sdkb.store import DiskStore, ReadPlan, Selection
from sdkb.trajectories import file_sha256


def linear_probe(pairs, worlds):
    unique = sorted(set(worlds))
    if len(unique) < 4:
        raise ValueError('At least four worlds required for a held-out probe')
    random.Random(7).shuffle(unique)
    training = set(unique[:len(unique) // 2])
    selected = torch.tensor([w in training for w in worlds])
    train = pairs[selected].reshape(-1, pairs.shape[-1]).double()
    test = pairs[~selected].reshape(-1, pairs.shape[-1]).double()
    mean, scale = train.mean(0), train.std(0).clamp_min(1e-6)
    train, test = (train - mean) / scale, (test - mean) / scale
    y_train = torch.tensor([-1., 1.], dtype=torch.float64).repeat(train.shape[0] // 2)
    y_test = torch.tensor([-1., 1.], dtype=torch.float64).repeat(test.shape[0] // 2)
    # Fixed regularization, not selected on the evaluation worlds.
    gram = train @ train.T / train.shape[1]
    alpha = torch.linalg.solve(gram + .01 * torch.eye(len(train), dtype=train.dtype), y_train)
    weight = train.T @ alpha / train.shape[1]
    prediction = test @ weight
    margins = prediction.reshape(-1, 2)[:, 1] - prediction.reshape(-1, 2)[:, 0]
    delta = pairs[:, 1] - pairs[:, 0]
    relative = delta.norm(dim=1) / pairs.norm(dim=2).mean(1).clamp_min(1e-12)
    return dict(train_worlds=len(training), test_worlds=len(unique) - len(training),
        train_accuracy=float(((train @ weight >= 0) == (y_train >= 0)).double().mean()),
        test_accuracy=float(((prediction >= 0) == (y_test >= 0)).double().mean()),
        positive_heldout_pair_margin_fraction=float((margins > 0).double().mean()),
        zero_heldout_pair_margin_fraction=float((margins == 0).double().mean()),
        pairs=len(pairs), payload_width=pairs.shape[-1],
        identical_serialized_pairs=int((delta == 0).all(1).sum()),
        mean_relative_bit_flip_norm=float(relative.mean()),
        regularization=.01, standardization='Training worlds only; float64 dual ridge')


def probe(episodes_file, directory):
    torch.set_num_threads(2)
    episodes = load_episodes(episodes_file)
    result = json.loads((directory / 'results.json').read_text())
    if {e.episode_id for e in episodes} != {r['episode'] for r in result['rows']}:
        raise ValueError('Episode identities differ from this evaluation')
    store = DiskStore(directory / 'bank.sqlite')
    sources = {}
    for e in episodes:
        for s in e.supports:
            value = (e.environment, s, e.query_time)
            if sources.setdefault(s.record_id, value) != value:
                raise ValueError('Conflicting source/world identity')
    output = {}
    for kind in ('permission', 'restoration'):
        pairs, worlds = [], []
        for world, source, query_time in sources.values():
            if source.kind != kind:
                continue
            match = re.search(r'capability=([01]);' if kind == 'permission' else r'(required|forbidden)\.$', source.text)
            if match is None:
                raise ValueError('Unknown generated rule format')
            bit = int(match[1]) if kind == 'permission' else int(match[1] == 'required')
            vectors = [store.fetch(ReadPlan(namespace, 's0', 'frozen-v1', 'research', query_time,
                       (Selection(source.record_id, 0.),)))[0].float()
                       for namespace in ('global', 'flip-' + kind)]
            pairs.append(torch.stack(vectors if bit == 0 else vectors[::-1]))
            worlds.append(world)
        output[kind] = linear_probe(torch.stack(pairs), worlds)
    return dict(episodes_sha256=file_sha256(episodes_file), evaluation=str(directory), probes=output,
                notice='A diagnostic trained readout, not deployed-model accuracy. '
                       'Counterfactual context paired; held-out worlds; no inference-time re-encoding.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(probe(args.episodes, args.evaluation), indent=2) + '\n')
