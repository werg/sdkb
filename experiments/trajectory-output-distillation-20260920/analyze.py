"""Verify the matched frozen-bank comparison and publish aggregate-only results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from sdkb.trajectories import file_sha256


CONDITIONS = ('all', 'zero_values', 'wrong_values', 'none')


def _episodes(report):
    by_episode = defaultdict(dict)
    for row in report['rows']:
        key = row['episode']
        if row['condition'] in by_episode[key]:
            raise ValueError('Duplicate episode condition')
        by_episode[key][row['condition']] = row
    if len(by_episode) != 119 or any(set(group) != set(CONDITIONS) for group in by_episode.values()):
        raise ValueError('Expected 119 complete four-condition episodes')
    for group in by_episode.values():
        original = group['all']['selected_ids']
        if original != group['zero_values']['selected_ids'] or original != group['wrong_values']['selected_ids']:
            raise ValueError('Payload intervention rerouted evidence')
        if len({row['token_count'] for row in group.values()}) != 1:
            raise ValueError('Token count differs across conditions')
    return by_episode


def _cluster_interval(values):
    groups = defaultdict(list)
    for trajectory, value in values:
        groups[trajectory].append(value)
    clusters = list(groups.values())
    rng = random.Random(233)
    draws = []
    for _ in range(2000):
        sample = [rng.choice(clusters) for _ in clusters]
        draws.append(sum(map(sum, sample)) / sum(map(len, sample)))
    draws.sort()
    return dict(mean_episode_delta=sum(map(sum, clusters)) / sum(map(len, clusters)),
                trajectory_groups=len(clusters), trajectory_bootstrap_interval_95=[draws[50], draws[1949]])


def analyze(root: Path, output: Path) -> dict:
    lock = json.loads((root / 'inputs.json').read_text())
    reports = {}
    for arm in ('control', 'distilled'):
        summary = json.loads((root / f'{arm}-depth' / 'summary.json').read_text())
        report_path = root / f'{arm}-depth' / 'depth-2.json'
        report = json.loads(report_path.read_text())
        if (summary['status'] != 'complete' or summary['protocol'] != 'teacher'
                or summary['writer_depth'] != 1 or not summary['same_serialized_bank']
                or [d['recurrence']['loops'] for d in summary['depths']] != [2]
                or report['recurrence']['loops'] != 2):
            raise ValueError('Incomplete fixed-bank R=2 teacher evaluation')
        identity = json.loads((root / f'{arm}-depth' / 'inputs.json').read_text())
        if identity['episodes_sha256'] != lock['validation_sha256']:
            raise ValueError('Validation set differs from declared study')
        if report['summary']['all']['episodes'] != 119:
            raise ValueError('Unexpected held-out count')
        reports[arm] = (report, _episodes(report), file_sha256(report_path),
                        file_sha256(root / f'{arm}-depth' / 'summary.json'))
    control, distilled = reports['control'][1], reports['distilled'][1]
    if set(control) != set(distilled):
        raise ValueError('Different held-out episodes across arms')
    for key in control:
        for condition in CONDITIONS:
            a, b = control[key][condition], distilled[key][condition]
            if (a['trajectory'], a['token_count'], a['support_annotation']) != (
                    b['trajectory'], b['token_count'], b['support_annotation']):
                raise ValueError('Matched episode identity changed')
            if a['selected_ids'] != b['selected_ids']:
                raise ValueError('Selected IDs changed across matched arms')
    comparisons = {}
    for condition in ('all', 'zero_values', 'wrong_values', 'none'):
        comparisons[f'control_to_distilled_{condition}_nll_reduction'] = _cluster_interval([
            (control[key][condition]['trajectory'],
             control[key][condition]['mean_nll'] - distilled[key][condition]['mean_nll'])
            for key in control])
    for control_name in ('zero_values', 'wrong_values'):
        comparisons[f'distillation_delta_real_over_{control_name}'] = _cluster_interval([
            (control[key]['all']['trajectory'],
             (distilled[key][control_name]['mean_nll'] - distilled[key]['all']['mean_nll'])
             - (control[key][control_name]['mean_nll'] - control[key]['all']['mean_nll']))
            for key in control])
    result = dict(protocol='119 repository-heldout SWE-smith episodes; separate frozen banks, same IDs and writer depth',
                  train_sha256=lock['train_sha256'], validation_sha256=lock['validation_sha256'],
                  parent_model_sha256=lock['parent_model_sha256'], arms={name: dict(
                      report_sha256=report_sha, summary_sha256=summary_sha,
                      writer_calls=json.loads((root / f'{name}-depth' / 'summary.json').read_text())['write_phase']['writer_calls'],
                      token_weighted_nll={condition: report['summary'][condition]['token_weighted_nll']
                                          for condition in CONDITIONS})
                      for name, (report, _, report_sha, summary_sha) in reports.items()},
                  paired=comparisons,
                  limits='Teacher-forced NLL and descriptive trajectory bootstrap, not agent success. Oracle selection and fixed plans do not establish learned retrieval. Text and latent paths have different token and compute budgets. Separate frozen banks reflect each arm’s trained writer; no cold-NVMe or capacity-substitution claim.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), indent=2))
