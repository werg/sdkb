"""Post-hoc format, position and old-vocabulary diagnostics; no new generation."""
import argparse
import json
from pathlib import Path
import random
import re

from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def analyze(root, source_corpus):
    inputs = json.loads((root/'inputs.json').read_text())
    if file_sha256(source_corpus) != inputs['checked_corpora']['source_finetuning']['sha256']:
        raise ValueError('Source finetuning corpus differs')
    source = {e.answer for e in EpisodeIndex(source_corpus) if e.task_family == 'multiuse/identifier'}
    vocabularies = {'source-heldout': (source, source)}
    for size, declared in inputs['data']['corpora'].items():
        episodes = EpisodeIndex(root/'data'/f'worlds-{size}.jsonl')
        if episodes.sha256 != declared['sha256']:
            raise ValueError('Continuation corpus changed')
        full = {e.answer for e in episodes if e.task_family == 'multiuse/identifier'}
        sampled, rng = set(), random.Random(inputs['seed'])
        for _step in range(inputs['steps']):
            rng.choice([2, 3])
            for _micro in range(inputs['data']['accumulation']):
                episode = rng.choice(episodes)
                if episode.task_family == 'multiuse/identifier':
                    sampled.add(episode.answer)
                for _source in episode.supports:
                    rng.random()
        if len(sampled) != declared['distinct_identifier_targets_sampled']:
            raise ValueError('Sampled vocabulary differs from the declared schedule')
        vocabularies[f'worlds-{size}'] = full, sampled
    reports = {}
    for label, (full, sampled) in vocabularies.items():
        path = root/'confirmation'/label/'results.json'
        result = json.loads(path.read_text())
        if result['inputs']['episodes_sha256'] != inputs['heldout_sha256']:
            raise ValueError('Confirmation corpus differs')
        conditions = {}
        for condition in sorted(result['generation_summary']['multiuse/identifier']):
            rows = [r for r in result['generation_rows']
                    if r['task_family'] == 'multiuse/identifier' and r['condition'] == condition]
            valid = [r for r in rows if re.fullmatch(r'api_[0-9a-f]{6}', r['prediction'])]
            positions = [sum(r['prediction'][4+p] == r['answer'][4+p] for r in valid) for p in range(6)]
            conditions[condition] = {'n': len(rows), 'valid_format': len(valid),
                'hex_positions_correct': positions, 'characters_correct': sum(positions),
                'characters': 6*len(rows),
                'predictions_in_full_training_target_vocabulary': sum(r['prediction'] in full for r in rows),
                'predictions_in_sampled_training_target_vocabulary': sum(r['prediction'] in sampled for r in rows)}
        reports[label] = {'results_sha256': file_sha256(path),
                         'full_training_target_vocabulary': len(full),
                         'sampled_training_target_vocabulary': len(sampled), 'conditions': conditions}
    atomic_json(root/'identifier-errors.json', {'arms': reports,
        'script_sha256': file_sha256(__file__),
        'notice': 'Post-hoc descriptive analysis of completed generations. Only full api_ plus six '
        'lowercase hexadecimal strings receive position credit; malformed outputs score zero. '
        'Sampled vocabulary is reconstructed only for continuation arms; the source column uses its '
        'complete finetuning vocabulary for both fields. Character agreement is not exact recall, '
        'and wrong predictions changing is not correct counterfactual behavior.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-corpus', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.source_corpus)
