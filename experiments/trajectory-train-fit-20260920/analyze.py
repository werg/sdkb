"""Compare frozen stored-payload teacher NLL on training and held-out trajectories."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from sdkb.checkpoints import resolve_checkpoint
from sdkb.trajectories import file_sha256


CONDITIONS = ('all', 'zero_values', 'wrong_values', 'none')


def _groups(report, count):
    groups = defaultdict(dict)
    for row in report['rows']:
        if row['condition'] in groups[row['episode']]:
            raise ValueError('Duplicate episode/condition')
        groups[row['episode']][row['condition']] = row
    if len(groups) != count or any(set(group) != set(CONDITIONS) for group in groups.values()):
        raise ValueError('Incomplete episode/condition matrix')
    for group in groups.values():
        real = group['all']
        for condition in CONDITIONS:
            row = group[condition]
            if (row['trajectory'], row['token_count']) != (real['trajectory'], real['token_count']):
                raise ValueError('Condition changed target identity or length')
            if condition != 'none' and row['selected_ids'] != real['selected_ids']:
                raise ValueError('Payload intervention changed selected sources')
    return groups


def _effect(groups, other):
    by_trajectory = defaultdict(list)
    for group in groups.values():
        real = group['all']
        by_trajectory[real['trajectory']].append(group[other]['mean_nll'] - real['mean_nll'])
    clusters = list(by_trajectory.values())
    rng = random.Random(233)
    draws = []
    for _ in range(2000):
        selected = [rng.choice(clusters) for _ in clusters]
        draws.append(sum(map(sum, selected)) / sum(map(len, selected)))
    draws.sort()
    return {'mean_episode_nll_benefit': sum(map(sum, clusters)) / len(groups),
            'trajectory_groups': len(clusters),
            'trajectory_bootstrap_interval_95': [draws[50], draws[1949]]}


def analyze(train_dir: Path, heldout_dir: Path, run: Path, train: Path,
            heldout: Path, output: Path):
    checkpoint = resolve_checkpoint(run, verify=True)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if manifest['dataset_sha256'] != file_sha256(train):
        raise ValueError('Training bytes differ from checkpoint manifest')
    checkpoint_hash = file_sha256(checkpoint / 'model.safetensors')
    datasets = {}
    all_groups = {}
    for name, directory, episodes, count in (
        ('train', train_dir, train, 530),
        ('heldout', heldout_dir, heldout, 119),
    ):
        identity = json.loads((directory / 'inputs.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        report_path = directory / 'depth-2.json'
        report = json.loads(report_path.read_text())
        if (identity['checkpoint_model_sha256'] != checkpoint_hash
                or identity['episodes_sha256'] != file_sha256(episodes)
                or identity['depths'] != [2] or identity['protocol'] != 'teacher'
                or identity['max_episodes'] != count or summary['status'] != 'complete'
                or summary['writer_depth'] != 1 or not summary['same_serialized_bank']
                or report['recurrence']['loops'] != 2):
            raise ValueError('Frozen-bank evaluator identity/protocol differs')
        groups = _groups(report, count)
        all_groups[name] = groups
        lengths = [group['all']['token_count'] for group in groups.values()]
        datasets[name] = {
            'episodes_sha256': identity['episodes_sha256'],
            'report_sha256': file_sha256(report_path),
            'summary_sha256': file_sha256(directory / 'summary.json'),
            'episodes': count,
            'trajectories': len({group['all']['trajectory'] for group in groups.values()}),
            'writer_calls': summary['write_phase']['writer_calls'],
            'target_tokens': sum(lengths),
            'mean_target_tokens': sum(lengths) / len(lengths),
            'token_weighted_nll': {condition: report['summary'][condition]['token_weighted_nll']
                                   for condition in CONDITIONS},
            'paired_episode_benefit': {other: _effect(groups, other)
                                       for other in ('zero_values', 'wrong_values', 'none')},
        }
    if set(all_groups['train']) & set(all_groups['heldout']):
        raise ValueError('Training and held-out episode IDs overlap')
    if ({group['all']['trajectory'] for group in all_groups['train'].values()}
            & {group['all']['trajectory'] for group in all_groups['heldout'].values()}):
        raise ValueError('Training and held-out trajectories overlap')
    result = {'protocol': 'Frozen control; full 530 training episodes versus 119 held-out episodes; R=2 stored-only teacher scoring',
              'analyzer_sha256': file_sha256(Path(__file__)),
              'checkpoint_model_sha256': checkpoint_hash,
              'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
              'datasets': datasets,
              'limits': 'Training-fit diagnostic only; unequal source/target distribution and target lengths may affect comparisons. Teacher NLL is not generation, controlled transfer or agent success.'}
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train_dir', 'heldout_dir', 'run', 'train', 'heldout', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(**vars(args)), indent=2))
