"""Verify complete frozen readout budgets and collect all declared controls."""
import argparse
import json
from pathlib import Path

import torch

from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def collect(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    if file_sha256(root/'episodes.jsonl') != inputs['episodes_sha256']:
        raise ValueError('Readout corpus changed')
    generator = torch.Generator().manual_seed(inputs['head_seed'])
    for _ in range(inputs['steps']):
        torch.randint(inputs['train_identifiers'], (inputs['batch'],), generator=generator)
    expected_sampler = generator.get_state()
    reports = {}
    for label, model in inputs['models'].items():
        bank = root/label/'bank/bank.sqlite'
        bank_hash = file_sha256(bank)
        reports[label] = {}
        for representation in inputs['representations']:
            reports[label][representation] = {}
            for head in inputs['heads']:
                stage = root/label/representation/head
                result = json.loads((stage/'results.json').read_text())
                identity = result['identity']
                state = torch.load(stage/'resume.pt', map_location='cpu', weights_only=True)
                if (state['identity'] != identity or state['step'] != inputs['steps']
                        or identity['source_manifest_sha256'] != model['manifest_sha256']
                        or identity['episodes_sha256'] != inputs['episodes_sha256']
                        or identity['bank_sha256'] != bank_hash
                        or not torch.equal(state['sampling_rng'], expected_sampler)):
                    raise ValueError(f'Incomplete or mismatched readout endpoint: {stage}')
                if identity['config']['train']['optimizer'] != 'muon':
                    raise ValueError('Readout optimizer differs')
                for group in state['optimizer']['param_groups']:
                    if group['lr'] != inputs['learning_rate'] or group['momentum'] != .95 or group['ns_steps'] != 5:
                        raise ValueError('Readout optimizer settings differ')
                rows = [json.loads(line) for line in (stage/'metrics.jsonl').read_text().splitlines()]
                if [row['step'] for row in rows] != list(range(20, inputs['steps']+1, 20)):
                    raise ValueError('Readout metrics do not cover the declared budget')
                if representation == 'zero_reader' and not all(identity['feature_constancy'].values()):
                    raise ValueError('Fixed-query zero-value features are not constant')
                for split, count in [('train', inputs['train_identifiers']), ('heldout', inputs['heldout_identifiers'])]:
                    for counts in result['results'][split].values():
                        if (counts['n'] != count or counts['characters'] != 6*count
                                or sum(counts['characters_correct_by_position']) != counts['characters_correct']):
                            raise ValueError('Readout result coverage or position totals differ')
                reports[label][representation][head] = {
                    'parameters': identity['parameters'], 'input_dimension': identity['input_dimension'],
                    'feature_constancy': identity['feature_constancy'], 'results': result['results'],
                    'resume_sha256': file_sha256(stage/'resume.pt'),
                    'results_sha256': file_sha256(stage/'results.json'),
                    'script_sha256': identity['script_sha256'], 'bank_sha256': bank_hash,
                    'all_declared_updates_and_final_sampler_verified': True,
                    'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'}
                                         for g in state['optimizer']['param_groups']],
                    'parameter_names': state['parameter_names']}
                del state
    atomic_json(output, {'inputs_sha256': file_sha256(root/'inputs.json'),
        'environment': json.loads((root/'environment.json').read_text()), 'arms': reports,
        'collector_sha256': file_sha256(__file__),
        'notice': 'All predeclared frozen models, representations and heads. One seed and an already '
                  'inspected held-out split. Supervised six-position format and unequal head sizes '
                  'are explicit. Readout accessibility is not language generation or an information bound.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
