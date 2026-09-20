"""Relate held-out text-context utility to stored-payload utility per episode."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from sdkb.trajectories import file_sha256


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, i in enumerate(order):
        ranks[i] = rank
    return ranks


def _correlation(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def _interval(groups, statistic, *, draws=4000):
    clusters = list(groups.values())
    rng = random.Random(233)
    samples = []
    for _ in range(draws):
        rows = [row for _ in clusters for row in rng.choice(clusters)]
        value = statistic(rows)
        if value is not None:
            samples.append(value)
    samples.sort()
    return [samples[int(.025 * len(samples))], samples[int(.975 * len(samples)) - 1]]


def analyze(root: Path, output: Path) -> dict:
    prior = json.loads((root / 'control-text-r1' / 'results.json').read_text())
    later = json.loads((root / 'distilled-text-r1' / 'results.json').read_text())
    if prior['rows'] != later['rows']:
        raise ValueError('One-pass text teacher changed between arms')
    text = defaultdict(dict)
    for row in prior['rows']:
        text[row['episode']][row['condition']] = row
    memory = {}
    for arm in ('control', 'distilled'):
        memory[arm] = defaultdict(dict)
        for row in json.loads((root / f'{arm}-depth' / 'depth-2.json').read_text())['rows']:
            memory[arm][row['episode']][row['condition']] = row
    if (len(text) != 119 or any(set(v) != {'r1_selected_text', 'r1_none'} for v in text.values())
            or any(set(memory[arm]) != set(text) for arm in memory)):
        raise ValueError('Incomplete matched validation episodes')
    episodes = []
    for episode, rows in text.items():
        values = [memory[arm][episode] for arm in ('control', 'distilled')]
        if any(set(v) != {'all', 'none', 'zero_values', 'wrong_values'} for v in values):
            raise ValueError('Missing payload control')
        if values[0]['all']['selected_ids'] != values[1]['all']['selected_ids']:
            raise ValueError('Selected sources differ between arms')
        episodes.append((rows['r1_none']['trajectory'],
                         rows['r1_none']['mean_nll'] - rows['r1_selected_text']['mean_nll'],
                         values[0]['zero_values']['mean_nll'] - values[0]['all']['mean_nll'],
                         values[1]['zero_values']['mean_nll'] - values[1]['all']['mean_nll']))
    groups = defaultdict(list)
    for row in episodes:
        groups[row[0]].append(row)
    def pearson_delta(rows):
        xs = [row[1] for row in rows]
        a = _correlation(xs, [row[2] for row in rows])
        b = _correlation(xs, [row[3] for row in rows])
        return b - a if a is not None and b is not None else None
    def spearman_delta(rows):
        xs = _rank([row[1] for row in rows])
        a = _correlation(xs, _rank([row[2] for row in rows]))
        b = _correlation(xs, _rank([row[3] for row in rows]))
        return b - a if a is not None and b is not None else None
    ordered = sorted(episodes, key=lambda row: row[1])
    quartiles = {}
    for name, subset in (('lowest_text_gain', ordered[:30]), ('highest_text_gain', ordered[-30:])):
        subset_groups = defaultdict(list)
        for row in subset:
            subset_groups[row[0]].append(row)
        quartiles[name] = dict(episodes=len(subset), trajectory_groups=len(subset_groups),
            mean_text_gain=statistics.mean(row[1] for row in subset),
            control_mean_payload_gain=statistics.mean(row[2] for row in subset),
            distilled_mean_payload_gain=statistics.mean(row[3] for row in subset),
            paired_payload_gain_delta=statistics.mean(row[3] - row[2] for row in subset),
            trajectory_bootstrap_interval_95=_interval(subset_groups,
                lambda rows: statistics.mean(row[3] - row[2] for row in rows)))
    xs = [row[1] for row in episodes]
    result = dict(protocol='119 matched repository-heldout episodes; source context effect vs real-over-zero payload effect',
        source_report_sha256={name: file_sha256(root / name) for name in (
            'control-text-r1/results.json', 'distilled-text-r1/results.json',
            'control-depth/depth-2.json', 'distilled-depth/depth-2.json')},
        analyzer_sha256=file_sha256(Path(__file__)), episodes=len(episodes), trajectory_groups=len(groups),
        control_pearson=_correlation(xs, [row[2] for row in episodes]),
        distilled_pearson=_correlation(xs, [row[3] for row in episodes]),
        pearson_delta=pearson_delta(episodes), pearson_delta_trajectory_interval_95=_interval(groups, pearson_delta),
        control_spearman=_correlation(_rank(xs), _rank([row[2] for row in episodes])),
        distilled_spearman=_correlation(_rank(xs), _rank([row[3] for row in episodes])),
        spearman_delta=spearman_delta(episodes), spearman_delta_trajectory_interval_95=_interval(groups, spearman_delta),
        quartiles=quartiles,
        limitations='Exploratory, post-hoc correlations on one model seed and 119 episodes; bootstrap clusters are trajectories, not independent training runs. Quartiles are selected using the observed text-context gain. A stronger correlation does not establish causal use, learned retrieval or agent success.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), indent=2))
