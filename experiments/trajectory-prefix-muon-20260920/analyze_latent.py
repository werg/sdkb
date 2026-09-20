"""Summarize the locked latent-stage held-out controls without publishing raw episodes."""
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
            raise ValueError('Incomplete paired condition')
        groups[pair[left]['trajectory']].append(pair[right]['mean_nll'] - pair[left]['mean_nll'])
    values, rng, samples = list(groups.values()), random.Random(197), []
    for _ in range(2000):
        selection = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, selection)) / sum(map(len, selection)))
    samples.sort()
    return dict(mean_episode_nll_reduction=sum(map(sum, values)) / sum(map(len, values)),
                trajectory_groups=len(values), trajectory_bootstrap_interval_95=[samples[50], samples[1949]],
                positive_episodes=sum(value > 0 for group in values for value in group),
                episodes=sum(map(len, values)))


def run(teacher_path, text_path, output):
    teacher = json.loads(teacher_path.read_text())
    text = json.loads(text_path.read_text())
    rows = teacher['rows']
    if (set(teacher['summary']) != {'all', 'none', 'zero_values', 'wrong_values'}
            or any(teacher['summary'][condition]['episodes'] != 119 for condition in teacher['summary'])
            or text['summary']['r2_selected_text']['episodes'] != 119):
        raise ValueError('Incomplete declared held-out conditions')
    groups = defaultdict(dict)
    for row in rows:
        groups[row['episode']][row['condition']] = row
    if len(groups) != 119 or any(
            pair['all']['selected_ids'] != pair['zero_values']['selected_ids']
            or pair['all']['selected_ids'] != pair['wrong_values']['selected_ids']
            for pair in groups.values()):
        raise ValueError('Payload controls changed the original read selection')
    report = dict(protocol='119 prepared repository-held-out public trajectory episodes; frozen stage-400 checkpoint',
                  teacher_results_sha256=sha256(teacher_path), text_results_sha256=sha256(text_path),
                  token_weighted_nll={key: value['token_weighted_nll'] for key, value in teacher['summary'].items()}
                  | {'r2_selected_text': text['summary']['r2_selected_text']['token_weighted_nll']},
                  paired_mean_episode_nll_reduction={
                      'real_values_over_zero_values': paired(rows, 'all', 'zero_values'),
                      'real_values_over_wrong_values': paired(rows, 'all', 'wrong_values'),
                      'real_values_over_no_memory': paired(rows, 'all', 'none')},
                  fixed_original_selection_episodes=len(groups),
                  resumed_writer_calls=teacher['write_phase']['writer_calls'],
                  stored_records=teacher['store']['records'],
                  serialized_payload_bytes=teacher['store']['serialized_payload_bytes'],
                  limits='Teacher-forced NLL only. Context length and compute differ across selected text, stored memory, and no-memory controls. Wrong-value permutation does not guarantee semantic disagreement; no agent success or capacity substitution claim.')
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.teacher, args.text, args.output), indent=2))
