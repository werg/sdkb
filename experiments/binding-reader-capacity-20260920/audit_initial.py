"""Verify initial non-reader tensors against the frozen source, without mutation."""
import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def audit(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    source = resolve_checkpoint(inputs['source'], verify=True)
    report = {'training_commit': None, 'arms': {}, 'inputs_sha256': file_sha256(root/'inputs.json')}
    for name in inputs['configs']:
        stage = root/name
        initial = list((stage/'checkpoints').glob('step-000000000-*'))
        if len(initial) != 1:
            raise ValueError('Expected one committed initial checkpoint')
        cp = resolve_checkpoint(initial[0], verify=True)
        provenance = json.loads((stage/'initialization.json').read_text())
        environment = json.loads((stage/'environment.json').read_text())
        if environment['git_dirty'] or not provenance['reader_reinitialized']:
            raise ValueError('Unexpected implementation/reset provenance')
        if report['training_commit'] not in (None, environment['git_commit']):
            raise ValueError('Arms used different implementations')
        report['training_commit'] = environment['git_commit']
        checked = []
        with safe_open(source/'model.safetensors', framework='pt', device='cpu') as before, safe_open(
                cp/'model.safetensors', framework='pt', device='cpu') as after:
            if set(before.keys()) != set(after.keys()):
                raise ValueError('Unexpected parameter names')
            reset_names = {key for key in after.keys() if key.startswith('reader.')}
            if reset_names != set(provenance['reinitialized_parameters']):
                raise ValueError('Reader reset provenance differs')
            for key in before.keys():
                if not key.startswith('reader.'):
                    if not torch.equal(before.get_tensor(key), after.get_tensor(key)):
                        raise ValueError(f'Non-reader tensor changed: {key}')
                    checked.append(key)
            if torch.equal(before.get_tensor('reader.initial_query.weight'), after.get_tensor('reader.initial_query.weight')):
                raise ValueError('Reader was not reset')
        optimizer = json.loads((stage/'optimizer.json').read_text())
        report['arms'][name] = {'initial_checkpoint': str(cp),
            'initial_manifest_sha256': file_sha256(cp/'manifest.json'),
            'initial_model_sha256': file_sha256(cp/'model.safetensors'),
            'all_non_reader_tensors_bitwise_equal': True, 'checked_tensor_names': checked,
            'reset_parameter_names': sorted(reset_names), 'environment': environment,
            'optimizer': optimizer}
    atomic_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)
