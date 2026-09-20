"""Verify completed training against the precommitted sampler and source identity."""
import argparse
import json
from pathlib import Path
import random
import runpy

import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json, run_lock
from sdkb.trajectories import file_sha256


def audit(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    launch = json.loads((root/'launch.json').read_text())
    helper = Path(__file__).parents[2]/'scripts/make_endpoint_freshness_control.py'
    if file_sha256(helper) != inputs['data']['script_sha256']:
        raise ValueError('Sampler implementation differs from the declared plan')
    sampling_plan = runpy.run_path(str(helper))['sampling_plan']
    reports = {}
    for size, corpus in inputs['data']['corpora'].items():
        stage = root/f'worlds-{size}'
        with run_lock(stage, clear_stop=False):
            checkpoint = resolve_checkpoint(stage, verify=True)
            manifest = json.loads((checkpoint/'manifest.json').read_text())
            if manifest['step'] != inputs['steps'] or manifest['dataset_sha256'] != corpus['sha256']:
                raise ValueError('Incomplete endpoint or changed dataset')
            episodes = EpisodeIndex(root/'data'/f'worlds-{size}.jsonl')
            if episodes.sha256 != corpus['sha256']:
                raise ValueError('Dataset changed after launch')
            plan = sampling_plan(episodes, seed=inputs['seed'], steps=inputs['steps'], accumulation=4)
            state = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=True)
            if (state['step'] != inputs['steps'] or state['optimizer_type'] != 'muon'
                    or state['python_rng'] != plan['final_rng_state'] or state.get('accumulation')):
                raise ValueError('Endpoint optimizer/sampler/accumulation differs')
            for key in ('schedule_sha256', 'families', 'distinct_identifier_targets_sampled',
                        'identifier_target_exposure_histogram'):
                if json.loads(json.dumps(plan[key])) != corpus[key]:
                    raise ValueError(f'Declared sampling differs: {key}')
            rows = [json.loads(line) for line in (stage/'metrics.jsonl').read_text().splitlines()]
            rng = random.Random(inputs['seed'])
            for step, row in enumerate(rows, 1):
                if row['step'] != step or row['loops'] != rng.choice([2, 3]):
                    raise ValueError('Actual optimizer steps or recurrent depths differ')
                for _micro in range(4):
                    episode = rng.choice(episodes)
                    for _source in episode.supports:
                        rng.random()
            if len(rows) != inputs['steps'] or rng.getstate() != state['python_rng']:
                raise ValueError('Metrics do not cover the complete declared schedule')
            environment = json.loads((stage/'environment.json').read_text())
            if environment['git_dirty'] or not environment['git_commit'].startswith(launch['training_commit']):
                raise ValueError('Training implementation differs from the frozen launch')
            initial = launch['arms'][size]
            memory = [row['memory'] for row in rows if 'memory' in row]
            reports[size] = {
                'checkpoint': str(checkpoint), 'manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                'model_sha256': manifest['sha256']['model.safetensors'],
                'initial_model_sha256': initial['initial_model_sha256'],
                'step': state['step'], 'all_depths_and_sampler_match_declared_plan': True,
                'elapsed_training_seconds': rows[-1]['elapsed_seconds'],
                'peak_logged_cuda_allocated_bytes': max(m['cuda_attempt_peak_allocated_bytes'] for m in memory),
                'minimum_logged_host_available_bytes': min(m['host_available_bytes'] for m in memory),
                'retained_checkpoint_directories': sorted(p.name for p in (stage/'checkpoints').iterdir()
                                                          if p.is_dir()),
            }
            del state
    atomic_json(output, {'training_commit': launch['training_commit'], 'arms': reports,
        'audit_script_sha256': file_sha256(__file__),
        'notice': 'Training provenance only; generation and intervention results are separate. '
                  'Contended time is not a throughput comparison.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
