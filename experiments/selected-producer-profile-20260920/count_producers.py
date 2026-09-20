"""Count source forward work from the verified native sampler, excluding replay."""
import argparse
import json
from pathlib import Path
import random

import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import evidence_ids
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json
from sdkb.training import config_from_run
from sdkb.trajectories import file_sha256


def count(root):
    inputs = json.loads((root/'inputs.json').read_text())
    checkpoint = resolve_checkpoint(root/'reference', verify=True)
    config = config_from_run(checkpoint)
    if config.train.live_fraction != 1 or config.memory.compaction != 'none' or not config.train.loop_counts:
        raise ValueError('Counter is specific to the declared all-live sampled-depth profile')
    episodes = EpisodeIndex(config.train.episodes_file)
    if episodes.sha256 != inputs['episodes_sha256']:
        raise ValueError('Corpus changed')
    rng = random.Random(config.train.seed)
    all_sources = selected_sources = 0
    for _ in range(config.train.steps):
        rng.choice(config.train.loop_counts)
        for _ in range(config.train.gradient_accumulation):
            episode = rng.choice(episodes)
            all_sources += len(episode.supports)
            selected_sources += len(evidence_ids(episode, config.train.evidence_scope))
            for _ in episode.supports:
                rng.random()
    state = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=True)
    if state['python_rng'] != rng.getstate() or state['step'] != inputs['steps']:
        raise ValueError('Actual endpoint sampler differs')
    report = {'scope': 'Forward source encodings derived from the verified actual sampler and pinned corpus; '
                       'excludes producer replays and decoder/reader work.',
        'reference_source_forwards': all_sources, 'selected_source_forwards': selected_sources,
        'omitted_unread_source_forwards': all_sources-selected_sources, 'sampler_matches_native_endpoint': True,
        'steps': config.train.steps, 'microbatches': config.train.steps*config.train.gradient_accumulation,
        'script_sha256': file_sha256(__file__), 'episodes_sha256': episodes.sha256}
    atomic_json(root/'producer-counts.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    count(parser.parse_args().root)
