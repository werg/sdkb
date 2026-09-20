"""Seal a common training-corpus diagnostic without consulting model outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

from sdkb.data import evidence_ids, load_episodes, save_episodes
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def prepare(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    if output.exists():
        raise FileExistsError(output)
    small = load_episodes(root/'data/worlds-256.jsonl')
    worlds = list(dict.fromkeys(e.environment for e in small))[:32]
    chosen = [e for e in small if e.environment in worlds
              and e.task_family == 'multiuse/identifier'
              and e.provenance.get('endpoint_character_position') is None]
    if len(chosen) != 64 or len({e.answer for e in chosen}) != 64:
        raise ValueError('Expected two distinct endpoint questions in each of 32 worlds')
    reports = {}
    for size, declaration in inputs['data']['corpora'].items():
        episodes = EpisodeIndex(root/'data'/f'worlds-{size}.jsonl')
        if episodes.sha256 != declaration['sha256']:
            raise ValueError('Training corpus changed')
        subset = {e.episode_id: e for e in episodes
                  if e.episode_id in {c.episode_id for c in chosen}}
        if any(subset.get(e.episode_id) != e for e in chosen):
            raise ValueError('Common diagnostic does not preserve both corpora exactly')
        rng = random.Random(inputs['seed'])
        targets, records = Counter(), Counter()
        for _step in range(inputs['steps']):
            rng.choice([2, 3])
            for _micro in range(inputs['data']['accumulation']):
                episode = rng.choice(episodes)
                if episode.task_family == 'multiuse/identifier':
                    targets[episode.answer] += 1
                    for record in evidence_ids(episode, 'required'):
                        records[record] += 1
                for _source in episode.supports:
                    rng.random()
        counts = []
        for episode in chosen:
            required = evidence_ids(episode, 'required')
            if len(required) != 1:
                raise ValueError('Diagnostic must select one prior endpoint record')
            counts.append({'episode': episode.episode_id,
                           'identifier_target_exposures': targets[episode.answer],
                           'identifier_selected_record_exposures': records[next(iter(required))]})
        reports[size] = {'corpus_sha256': episodes.sha256, 'rows': counts,
                        'target_exposure_histogram': dict(Counter(
                            row['identifier_target_exposures'] for row in counts))}
    output.mkdir()
    path = output/'episodes.jsonl'
    save_episodes(path, chosen)
    if load_episodes(path) != chosen:
        raise ValueError('Diagnostic serialization changed source or query contents')
    report = {'script_sha256': file_sha256(__file__), 'episodes_sha256': file_sha256(path),
              'episodes': len(chosen), 'worlds': len(worlds), 'arms': reports,
              'selection': 'First 32 training worlds in corpus order; both original whole-endpoint '
                           'questions, excluding six replica questions. Identical serialized episodes '
                           'are present in both corpora. No model outcomes used.',
              'notice': 'Supplemental training-corpus diagnostic declared while both trainings run. '
                        'Not held-out evidence. Exposure counts cover endpoint-target microbatches; '
                        'they do not count appearances in action/rule source text or teacher anchors. '
                        'Full held-out confirmation remains primary.'}
    atomic_json(output/'inputs.json', report)
    print(json.dumps({k: v['target_exposure_histogram'] for k, v in reports.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.output)
