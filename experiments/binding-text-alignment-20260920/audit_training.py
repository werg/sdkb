"""Verify warm-start identity, complete optimizer state and the actual sampling budget."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch
from safetensors.torch import load_file

from sdkb.checkpoints import resolve_checkpoint
from sdkb.config import load_config
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json, run_lock
from sdkb.training import config_from_run
from sdkb.trajectories import file_sha256


def audit(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    launch = json.loads((root/'launch.json').read_text())
    source = resolve_checkpoint(inputs['source'], verify=True)
    if file_sha256(source/'manifest.json') != inputs['source_manifest_sha256']:
        raise ValueError('Warm-start source changed')
    source_weights = load_file(str(source/'model.safetensors'))
    episodes = EpisodeIndex(inputs['episodes'])
    if episodes.sha256 != inputs['episodes_sha256']:
        raise ValueError('Corpus changed')
    reports = {}
    for name, config_sha in inputs['configs'].items():
        stage = root/name
        with run_lock(stage, clear_stop=False):
            declared = root/(name+'.yaml')
            checkpoint = resolve_checkpoint(stage, verify=True)
            config = config_from_run(checkpoint)
            if (file_sha256(declared) != config_sha or asdict(config) != asdict(load_config(declared))
                    or config.train.reinitialize_reader or config.memory.compaction != 'none'
                    or config.train.live_fraction != 1. or not config.train.selected_producers_only):
                raise ValueError('Training configuration changed')
            initial = [p.parent for p in (stage/'checkpoints').glob('*/manifest.json')
                       if json.loads(p.read_text())['step'] == 0]
            if len(initial) != 1:
                raise ValueError('Expected one retained initial checkpoint')
            initial = resolve_checkpoint(initial[0], verify=True)
            weights = load_file(str(initial/'model.safetensors'))
            if weights.keys() != source_weights.keys() or any(
                    not torch.equal(weights[k], source_weights[k]) for k in weights):
                raise ValueError('Initial weights differ from common source')
            del weights
            state = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=True)
            manifest = json.loads((checkpoint/'manifest.json').read_text())
            if (state['step'] != inputs['steps'] or state['optimizer_type'] != 'muon'
                    or state.get('accumulation') or manifest['step'] != inputs['steps']
                    or manifest['dataset_sha256'] != episodes.sha256):
                raise ValueError('Incomplete endpoint')
            rows = [json.loads(line) for line in (stage/'metrics.jsonl').read_text().splitlines()]
            rng = random.Random(config.train.seed)
            for step, row in enumerate(rows, 1):
                if row['step'] != step or row['loops'] != rng.choice(config.train.loop_counts):
                    raise ValueError('Actual update/depth schedule differs')
                for _ in range(config.train.gradient_accumulation):
                    episode = rng.choice(episodes)
                    for _ in episode.supports:
                        rng.random()
                expected = (row['loss'] + config.train.oracle_anchor_weight*row['oracle_anchor_nll']
                            + config.train.oracle_alignment_weight*row['oracle_alignment_loss'])
                if abs(expected-row['optimization_loss']) > 1e-9:
                    raise ValueError('Optimization loss accounting differs')
            if len(rows) != inputs['steps'] or rng.getstate() != state['python_rng']:
                raise ValueError('Incomplete sampler budget')
            environment = json.loads((stage/'environment.json').read_text())
            if environment['git_dirty'] or environment['git_commit'] != launch['training_commit']:
                raise ValueError('Implementation changed')
            memory = [r['memory'] for r in rows if 'memory' in r]
            reports[name] = dict(checkpoint=str(checkpoint),
                manifest_sha256=file_sha256(checkpoint/'manifest.json'),
                initial_manifest_sha256=file_sha256(initial/'manifest.json'),
                model_sha256=manifest['sha256']['model.safetensors'],
                all_initial_weights_equal_source=True, all_depths_and_sampler_verified=True,
                complete_muon_state=True, steps=state['step'],
                metrics_sha256=file_sha256(stage/'metrics.jsonl'),
                elapsed_training_seconds=rows[-1]['elapsed_seconds'],
                peak_logged_cuda_allocated_bytes=max(m['cuda_attempt_peak_allocated_bytes'] for m in memory),
                minimum_logged_host_available_bytes=min(m['host_available_bytes'] for m in memory))
    atomic_json(output, dict(arms=reports, inputs_sha256=file_sha256(root/'inputs.json'),
                            auditor_sha256=file_sha256(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
