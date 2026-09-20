"""Audit completed width forks against their common deterministic schedule."""
import argparse
import json
from pathlib import Path
import random
import runpy

import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.training import config_from_run
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json, run_lock
from sdkb.trajectories import file_sha256


def audit(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    launch = json.loads((root/'launch.json').read_text())
    code = Path(__file__).parents[2]
    episodes = EpisodeIndex(inputs['episodes'])
    if episodes.sha256 != inputs['episodes_sha256']:
        raise ValueError('Training corpus changed')
    helper = code/'scripts/make_endpoint_freshness_control.py'
    plan = runpy.run_path(str(helper))['sampling_plan'](episodes, seed=inputs['seed'],
                                                       steps=inputs['steps'], accumulation=4)
    reports = {}
    for name in inputs['configs']:
        stage = root/name
        with run_lock(stage, clear_stop=False):
            checkpoint = resolve_checkpoint(stage, verify=True)
            manifest = json.loads((checkpoint/'manifest.json').read_text())
            config = config_from_run(checkpoint)
            if (manifest['step'] != inputs['steps'] or manifest['dataset_sha256'] != episodes.sha256
                    or not config.train.reinitialize_reader or config.memory.reader_width != int(name.split('-')[1])):
                raise ValueError('Incomplete or changed endpoint')
            state = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=True)
            if (state['step'] != inputs['steps'] or state['optimizer_type'] != 'muon'
                    or state['python_rng'] != plan['final_rng_state'] or state.get('accumulation')):
                raise ValueError('Endpoint optimizer/sampler/accumulation differs')
            rows = [json.loads(line) for line in (stage/'metrics.jsonl').read_text().splitlines()]
            rng = random.Random(inputs['seed'])
            for step, row in enumerate(rows, 1):
                if row['step'] != step or row['loops'] != rng.choice([2, 3]):
                    raise ValueError('Actual updates/depths differ')
                for _ in range(4):
                    episode = rng.choice(episodes)
                    for _ in episode.supports:
                        rng.random()
            if len(rows) != inputs['steps'] or rng.getstate() != state['python_rng']:
                raise ValueError('Incomplete actual sampling schedule')
            environment = json.loads((stage/'environment.json').read_text())
            if environment['git_dirty'] or environment['git_commit'] != launch['training_commit']:
                raise ValueError('Training implementation changed')
            memory = [row['memory'] for row in rows if 'memory' in row]
            reports[name] = {'checkpoint': str(checkpoint),
                'manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                'model_sha256': manifest['sha256']['model.safetensors'], 'steps': state['step'],
                'all_depths_and_sampler_verified': True,
                'elapsed_training_seconds': rows[-1]['elapsed_seconds'],
                'peak_logged_cuda_allocated_bytes': max(m['cuda_attempt_peak_allocated_bytes'] for m in memory),
                'minimum_logged_host_available_bytes': min(m['host_available_bytes'] for m in memory),
                'checkpoint_directories': sorted(p.name for p in (stage/'checkpoints').iterdir() if p.is_dir())}
            del state
    atomic_json(output, {'training_commit': launch['training_commit'], 'arms': reports,
        'common_sampling_plan': {k: v for k, v in plan.items() if k != 'final_rng_state'},
        'sampler_helper_sha256': file_sha256(helper), 'audit_script_sha256': file_sha256(__file__),
        'notice': 'Training provenance and contended resource telemetry; held-out behavior is separate.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
