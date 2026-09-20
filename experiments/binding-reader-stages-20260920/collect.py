"""Audit complete intermediate readouts, including sampler and optimizer identity."""
import argparse
import json
from pathlib import Path

import torch

from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def collect(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    for key in ('episodes', 'bank'):
        if file_sha256(inputs[key]) != inputs[key+'_sha256']:
            raise ValueError('Diagnostic input changed')
    generator = torch.Generator().manual_seed(inputs['seed'])
    for _ in range(inputs['steps']):
        torch.randint(1984, (inputs['batch'],), generator=generator)
    reports = {}
    for representation in inputs['representations']:
        reports[representation] = {}
        for head in inputs['heads']:
            stage = root/representation/head
            result = json.loads((stage/'results.json').read_text())
            identity = result['identity']
            state = torch.load(stage/'resume.pt', map_location='cpu', weights_only=True)
            if (state['identity'] != identity or state['step'] != inputs['steps']
                    or identity['source_manifest_sha256'] != inputs['manifest_sha256']
                    or identity['episodes_sha256'] != inputs['episodes_sha256']
                    or identity['bank_sha256'] != inputs['bank_sha256']
                    or identity['script_sha256'] != inputs['probe_script_sha256']
                    or not torch.equal(state['sampling_rng'], generator.get_state())):
                raise ValueError(f'Incomplete or changed endpoint: {stage}')
            if identity['config']['train']['optimizer'] != 'muon':
                raise ValueError('Optimizer differs')
            for group in state['optimizer']['param_groups']:
                if (group['lr'] != .001 or group['momentum'] != .95 or group['ns_steps'] != 5
                        or group['weight_decay'] != .01 or group['adjust_lr_fn'] != 'match_rms_adamw'):
                    raise ValueError('Muon settings differ')
            rows = [json.loads(line) for line in (stage/'metrics.jsonl').read_text().splitlines()]
            if [row['step'] for row in rows] != list(range(20, inputs['steps']+1, 20)):
                raise ValueError('Incomplete metric budget')
            for split, count in [('train', 1984), ('heldout', 64)]:
                for counts in result['results'][split].values():
                    if (counts['n'] != count or counts['characters'] != 6*count
                            or sum(counts['characters_correct_by_position']) != counts['characters_correct']):
                        raise ValueError('Result coverage differs')
            reports[representation][head] = {'identity': {key: identity[key] for key in (
                'parameters', 'input_dimension', 'feature_capture_dtypes', 'feature_constancy',
                'source_manifest_sha256', 'episodes_sha256', 'bank_sha256', 'script_sha256')},
                'results': result['results'],
                'resume_sha256': file_sha256(stage/'resume.pt'), 'results_sha256': file_sha256(stage/'results.json'),
                'complete_budget_and_sampler_verified': True,
                'optimizer_groups': [{k: v for k, v in group.items() if k != 'params'}
                                     for group in state['optimizer']['param_groups']]}
    atomic_json(output, {'inputs_sha256': file_sha256(root/'inputs.json'), 'arms': reports,
        'environment': json.loads((root/'environment.json').read_text()),
        'collector_sha256': file_sha256(__file__), 'notice': inputs['notice']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
