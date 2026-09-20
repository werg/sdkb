"""Compare complete native producer-policy endpoints without relaxing precision."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from safetensors.torch import load_file
import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.config import load_config
from sdkb.training import config_from_run
from sdkb.operations import atomic_json, run_lock
from sdkb.trajectories import file_sha256


def differences(a, b, path='root'):
    if isinstance(a, torch.Tensor):
        return [] if isinstance(b, torch.Tensor) and torch.equal(a, b) else [path]
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return [path+'.keys']
        return [item for key in a for item in differences(a[key], b[key], path+'.'+str(key))]
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return [path+'.length_or_type']
        return [item for i, (left, right) in enumerate(zip(a, b))
                for item in differences(left, right, path+'.'+str(i))]
    return [] if a == b else [path]


def collect(root, output):
    with run_lock(root, clear_stop=False):
        inputs = json.loads((root/'inputs.json').read_text())
        reports, states, weights, metrics = {}, {}, {}, {}
        for name in inputs['order']:
            stage = root/name
            with run_lock(stage, clear_stop=False):
                checkpoint = resolve_checkpoint(stage, verify=True)
                manifest = json.loads((checkpoint/'manifest.json').read_text())
                if manifest['step'] != inputs['steps'] or manifest['dataset_sha256'] != inputs['episodes_sha256']:
                    raise ValueError('Incomplete or changed native profile')
                declared = inputs['configs'][name]
                if (file_sha256(declared['config']) != declared['sha256']
                        or asdict(config_from_run(checkpoint)) != asdict(load_config(declared['config']))):
                    raise ValueError('Saved configuration differs from declaration')
                states[name] = torch.load(checkpoint/'training_state.pt', map_location='cpu', weights_only=True)
                if states[name]['optimizer_type'] != 'muon' or states[name]['step'] != inputs['steps']:
                    raise ValueError('Incomplete Muon state')
                weights[name] = load_file(str(checkpoint/'model.safetensors'))
                rows = [json.loads(line) for line in (stage/'metrics.jsonl').read_text().splitlines()]
                if [r['step'] for r in rows] != list(range(1, inputs['steps']+1)):
                    raise ValueError('Incomplete metric budget')
                metrics[name] = [{k: v for k, v in row.items() if k not in {'elapsed_seconds', 'memory'}} for row in rows]
                deltas = [b['elapsed_seconds']-a['elapsed_seconds'] for a, b in zip(rows[4:-1], rows[5:])]
                environment = json.loads((stage/'environment.json').read_text())
                if environment['git_dirty']:
                    raise ValueError('Profile code was not frozen')
                reports[name] = {'checkpoint': str(checkpoint),
                    'manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                    'model_sha256': manifest['sha256']['model.safetensors'],
                    'environment': environment, 'completed_updates': len(rows),
                    'median_update_seconds_after_first_five': statistics.median(deltas),
                    'elapsed_training_seconds': rows[-1]['elapsed_seconds'],
                    'peak_logged_cuda_allocated_bytes': max(row['memory']['cuda_attempt_peak_allocated_bytes'] for row in rows)}
        if reports['reference']['environment']['git_commit'] != reports['selected']['environment']['git_commit']:
            raise ValueError('Profile implementations differ')
        diff = {'model': differences(weights['reference'], weights['selected']),
                'optimizer_and_all_rng': differences(states['reference'], states['selected']),
                'all_training_metrics': differences(metrics['reference'], metrics['selected'])}
        atomic_json(output, {'arms': reports, 'difference_paths': diff,
            'all_model_optimizer_rng_and_training_metrics_bitwise_equal': not any(diff.values()),
            'observed_median_selected_over_reference_time_ratio': reports['selected']['median_update_seconds_after_first_five']/reports['reference']['median_update_seconds_after_first_five'],
            'inputs_sha256': file_sha256(root/'inputs.json'), 'collector_sha256': file_sha256(__file__),
            'notice': 'Native equivalence on this deterministic-writer fixture only. Timings are sequential and contended; no isolated throughput or capability claim.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
