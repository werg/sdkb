"""Post-hoc positional identifier accuracy; malformed/missing positions are wrong."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


def correct_positions(row):
    answer, prediction = row['answer'], row['prediction']
    if not re.fullmatch(r'api_[0-9a-f]{6}', answer):
        raise ValueError('Unexpected target format')
    return [int(prediction.startswith('api_') and len(prediction) > i
                and prediction[i] == answer[i]) for i in range(4, 10)]


def analyze(path):
    data = json.loads(path.read_text())
    rows = [r for r in data['generation_rows'] if r['task_family'] == 'multiuse/identifier']
    grouped = defaultdict(list)
    indexed = {}
    for row in rows:
        key = row['episode'], row['condition']
        if key in indexed:
            raise ValueError('Duplicate episode/condition')
        indexed[key] = row
        grouped[row['condition']].append(row)
    conditions = {}
    for condition, items in grouped.items():
        positions = [correct_positions(r) for r in items]
        counts = [sum(p[i] for p in positions) for i in range(6)]
        conditions[condition] = {
            'queries': len(items), 'positions': 6 * len(items),
            'correct_positions': sum(counts), 'correct_by_position': counts,
            'well_formed': sum(bool(re.fullmatch(r'api_[0-9a-f]{6}', r['prediction'])) for r in items),
            'unique_predictions': len({r['prediction'] for r in items}),
        }
    worlds = defaultdict(list)
    for row in grouped['all']:
        zero = indexed[row['episode'], 'zero_values']
        if (row['environment'], row['answer']) != (zero['environment'], zero['answer']):
            raise ValueError('Ablation pairing changed')
        worlds[row['environment']].append(sum(correct_positions(row)) - sum(correct_positions(zero)))
    values = list(worlds.values())
    rng, samples = random.Random(107), []
    for _ in range(5000):
        sample = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, sample)) / (6 * sum(map(len, sample))))
    samples.sort()
    return {
        'results_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'conditions': conditions,
        'all_minus_zero_character_accuracy': sum(map(sum, values)) / (6 * sum(map(len, values))),
        'world_bootstrap_95_percentile_interval': [samples[125], samples[4874]],
        'bootstrap_worlds': len(values), 'bootstrap_samples': 5000, 'bootstrap_seed': 107,
        'notice': 'Post-hoc, one training seed; intervals do not account for multiple comparisons or seed variance. '
                  'Missing/malformed positions count as wrong. No alignment or edit-distance credit.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {arm: analyze(args.root/arm/'results.json')
              for arm in ('source', 'independent-worlds', 'views')}
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
