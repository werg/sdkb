"""Prepare fixed-query readouts of all three completed freshness-study models."""
import argparse
import json
from pathlib import Path
import runpy

from sdkb.archiving import ensure_free
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes, make_multiuse_world, save_episodes
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


TRAIN_SPLIT = 'freshness-readout-train-20260920'


def make_examples(heldout, excluded_targets, excluded_sources, *, training_worlds=992):
    if training_worlds < 1:
        raise ValueError('Positive training world count required')
    questions = runpy.run_path(str(Path(__file__).parents[2]/'scripts/make_identifier_character_control.py'))
    forbidden = set(excluded_targets)
    source_ids = set(excluded_sources)
    for episode in heldout:
        source_ids.update(s.record_id for s in episode.supports)
        if any(s.created_at >= episode.query_time for s in episode.supports):
            raise ValueError('Readout sources must precede their queries')
        if episode.task_family == 'multiuse/identifier':
            questions['validate_identifier'](episode)
            if episode.query != questions['FULL_QUERY']:
                raise ValueError('Every identifier query must be identical')
            forbidden.add(episode.answer)
            forbidden.add('api_'+''.join(format(int(c, 16)^15, 'x') for c in episode.answer[4:]))
    examples, accepted, rejected = list(heldout), [], []
    for seed in range(10*training_worlds):
        world = make_multiuse_world(seed, split=TRAIN_SPLIT, bindings=2)
        identifiers = [e for e in world if e.task_family == 'multiuse/identifier']
        if ({e.answer for e in identifiers} & forbidden
                or {s.record_id for e in world for s in e.supports} & source_ids):
            rejected.append(seed)
            continue
        examples.extend(questions['full_episode'](e) if e.task_family == 'multiuse/identifier' else e
                        for e in world)
        accepted.append(seed)
        if len(accepted) == training_worlds:
            break
    if len(accepted) != training_worlds:
        raise ValueError('Insufficient disjoint worlds within the declared search budget')
    return examples, {'accepted_seeds': accepted, 'rejected_seeds': rejected,
                      'excluded_endpoint_strings': len(forbidden), 'excluded_source_ids': len(source_ids)}


def prepare(study, source_corpus, output):
    inputs = json.loads((study/'inputs.json').read_text())
    audit = json.loads((study/'endpoint-audit.json').read_text())
    if file_sha256(study/'heldout.jsonl') != inputs['heldout_sha256']:
        raise ValueError('Original held-out corpus changed')
    paths = [(source_corpus, inputs['checked_corpora']['source_finetuning']['sha256']),
             (study/'data/worlds-8192.jsonl', inputs['data']['corpora']['8192']['sha256'])]
    targets, source_ids = set(), set()
    for path, digest in paths:
        episodes = EpisodeIndex(path)
        if episodes.sha256 != digest:
            raise ValueError('Writer-training corpus changed')
        for episode in episodes:
            source_ids.update(s.record_id for s in episode.supports)
            if episode.task_family == 'multiuse/identifier':
                targets.add(episode.answer)
    heldout = load_episodes(study/'heldout.jsonl')
    examples, sampling = make_examples(heldout, targets, source_ids)
    models = {'source': {'checkpoint': inputs['source'],
                         'manifest_sha256': inputs['source_manifest_sha256']},
              **{f'worlds-{s}': {'checkpoint': a['checkpoint'], 'manifest_sha256': a['manifest_sha256']}
                 for s, a in audit['arms'].items()}}
    for model in models.values():
        checkpoint = resolve_checkpoint(model['checkpoint'], verify=True)
        if file_sha256(checkpoint/'manifest.json') != model['manifest_sha256']:
            raise ValueError('Frozen model identity changed')
    ensure_free(output.parent, 64*1024**2, 10*1024**3)
    output.mkdir(exist_ok=False)
    corpus = output/'episodes.jsonl'
    save_episodes(corpus, examples)
    if load_episodes(corpus)[:len(heldout)] != heldout:
        raise ValueError('Held-out episodes changed during serialization')
    record = {'models': models, 'episodes_sha256': file_sha256(corpus),
              'heldout_sha256': inputs['heldout_sha256'], 'heldout_worlds': 32,
              'train_worlds': 992, 'train_identifiers': 1984, 'heldout_identifiers': 64,
              'steps': 1600, 'batch': 128, 'head_seed': 83, 'learning_rate': .001,
              'representations': ['payload', 'reader', 'zero_reader'], 'heads': ['linear', 'mlp256'],
              'sampling': sampling, 'script_sha256': file_sha256(__file__),
              'question_generator_sha256': file_sha256(Path(__file__).parents[2]/'scripts/make_identifier_character_control.py'),
              'excluded_corpora': {str(p): h for p, h in paths},
              'notice': 'Post-hoc diagnostic using the already inspected sealed 32 worlds. '
                        'Head-training worlds exclude prior writer-training source IDs and endpoint targets, '
                        'plus original/inverted held-out endpoints. Identifier queries are identical; '
                        'supervised six-position heads supply output format and are not language generation.'}
    atomic_json(output/'inputs.json', record)
    print(json.dumps({k: record[k] for k in ('train_worlds', 'train_identifiers', 'heldout_identifiers')}
                     | {'rejected_seeds': sampling['rejected_seeds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source-corpus', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.study, args.source_corpus, args.output)
