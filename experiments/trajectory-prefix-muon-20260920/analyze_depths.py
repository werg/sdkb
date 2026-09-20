"""Summarize same-bank latent recurrence controls without raw trajectories."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def paired(rows, left, right):
    by_episode = defaultdict(dict)
    for row in rows:
        by_episode[row['episode']][row['condition']] = row
    groups = defaultdict(list)
    for pair in by_episode.values():
        if left not in pair or right not in pair:
            raise ValueError('Incomplete depth condition')
        groups[pair[left]['trajectory']].append(pair[right]['mean_nll'] - pair[left]['mean_nll'])
    values, rng, samples = list(groups.values()), random.Random(197), []
    for _ in range(2000):
        selection = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, selection)) / sum(map(len, selection)))
    samples.sort()
    return dict(mean_episode_nll_reduction=sum(map(sum, values)) / sum(map(len, values)),
                trajectory_groups=len(values), trajectory_bootstrap_interval_95=[samples[50], samples[1949]],
                positive_episodes=sum(value > 0 for group in values for value in group))


def run(sweep, first_text, later_text, output):
    summary_path = sweep/'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary['status'] != 'complete' or not summary['same_serialized_bank']:
        raise ValueError('Incomplete or mismatched frozen depth sweep')
    text = json.loads(first_text.read_text())['summary'] | json.loads(later_text.read_text())['summary']
    reports = {}
    for depth in (1, 2, 3, 4):
        path = sweep/f'depth-{depth}.json'
        report = json.loads(path.read_text())
        if report['recurrence']['loops'] != depth or any(
                report['summary'][c]['episodes'] != 119 for c in ('all', 'none', 'zero_values', 'wrong_values')):
            raise ValueError('Incomplete declared depth')
        pairs = defaultdict(dict)
        for row in report['rows']:
            pairs[row['episode']][row['condition']] = row
        if len(pairs) != 119 or any(pair['all']['selected_ids'] != pair['zero_values']['selected_ids']
                or pair['all']['selected_ids'] != pair['wrong_values']['selected_ids'] for pair in pairs.values()):
            raise ValueError('Depth payload control rerouted evidence')
        reports[str(depth)] = dict(report_sha256=sha256(path),
            token_weighted_nll={c: report['summary'][c]['token_weighted_nll']
                                for c in ('all', 'none', 'zero_values', 'wrong_values')}
                | {'selected_text': text[f'r{depth}_selected_text']['token_weighted_nll']},
            real_over_zero=paired(report['rows'], 'all', 'zero_values'),
            real_over_wrong=paired(report['rows'], 'all', 'wrong_values'),
            fixed_selection_episodes=len(pairs),
            beyond_final_training_depth=report['depth_exceeds_final_training_max'])
    output_report = dict(protocol='119 repository-held-out teacher episodes; same frozen latent step-400 model and bank',
        sweep_sha256=sha256(summary_path), text_sha256=[sha256(first_text), sha256(later_text)],
        writer_calls=summary['write_phase']['writer_calls'], stored_records=summary['bank_sizes']['records'],
        depths=reports, limitations='Teacher NLL and descriptive trajectory bootstrap only. R=4 exceeds the latent stage training depth. Selected text, no memory and stored memory use different layouts/computation. OS cache state is uncontrolled; this is not cold-NVMe or serving latency. No agent success or parameter-substitution claim.')
    output.write_text(json.dumps(output_report, indent=2)+'\n')
    return output_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--first-text', type=Path, required=True)
    parser.add_argument('--later-text', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.sweep, args.first_text, args.later_text, args.output), indent=2))
