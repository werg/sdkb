"""Verify and compare the trained memory-gate fork with its uniform control."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random

from safetensors import safe_open

from sdkb.config import load_config
from sdkb.trajectories import file_sha256


CONDITIONS = ('all', 'zero_values', 'wrong_values', 'none')


def _group(report):
    groups = defaultdict(dict)
    for row in report['rows']:
        if row['condition'] in groups[row['episode']]:
            raise ValueError('Duplicate condition')
        groups[row['episode']][row['condition']] = row
    if len(groups) != 119 or any(set(group) != set(CONDITIONS) for group in groups.values()):
        raise ValueError('Incomplete 119-episode condition set')
    for group in groups.values():
        if (group['all']['selected_ids'] != group['zero_values']['selected_ids']
                or group['all']['selected_ids'] != group['wrong_values']['selected_ids']):
            raise ValueError('Payload intervention rerouted evidence')
    return groups


def _paired(control, gate, comparison):
    clusters = defaultdict(list)
    for episode in control:
        a, b = control[episode], gate[episode]
        if comparison in CONDITIONS:
            value = a[comparison]['mean_nll'] - b[comparison]['mean_nll']
        else:
            other = comparison.removeprefix('real_over_')
            value = ((b[other]['mean_nll'] - b['all']['mean_nll'])
                     - (a[other]['mean_nll'] - a['all']['mean_nll']))
        clusters[a['all']['trajectory']].append(value)
    groups = list(clusters.values())
    rng = random.Random(233)
    draws = []
    for _ in range(2000):
        selected = [rng.choice(groups) for _ in groups]
        draws.append(sum(map(sum, selected)) / sum(map(len, selected)))
    draws.sort()
    return dict(mean_episode_delta=sum(map(sum, groups)) / sum(map(len, groups)),
                trajectory_groups=len(groups), trajectory_bootstrap_interval_95=[draws[50], draws[1949]])


def analyze(root: Path, control_root: Path, output: Path) -> dict:
    lock = json.loads((root / 'inputs.json').read_text())
    control_lock = json.loads((control_root / 'inputs.json').read_text())
    if (lock['parent_model_sha256'] != control_lock['parent_model_sha256']
            or lock['train_sha256'] != control_lock['train_sha256']
            or lock['validation_sha256'] != control_lock['validation_sha256']):
        raise ValueError('Parent, training or validation inputs differ')
    if file_sha256(root / 'memory-gate.yaml') != lock['config_sha256']:
        raise ValueError('Gate config changed')
    a, b = [asdict(load_config(path)) for path in (control_root / 'control.yaml', root / 'memory-gate.yaml')]
    if a['train']['warmstart_memory_gate'] is not None or b['train']['warmstart_memory_gate'] != .30:
        raise ValueError('Gate intervention differs from declared policy')
    for config in (a, b):
        config['train'].pop('wandb_group')
        config['train'].pop('warmstart_memory_gate')
    if a != b:
        raise ValueError('Non-gate architecture or training settings differ')
    provenance = json.loads((root / 'memory_gate' / 'initialization.json').read_text())
    if provenance['warmstart_memory_gate'] != .30 or provenance['prior_memory_gate'] is None:
        raise ValueError('Gate override was not recorded at warm-start')
    initial = next((root / 'memory_gate' / 'checkpoints').glob('step-000000000-*'))
    with safe_open(initial / 'model.safetensors', framework='pt', device='cpu') as tensors:
        initial_gate = float(tensors.get_tensor('backbone.bridge.memory_logit').sigmoid())
    if abs(initial_gate - .30) > 1e-6:
        raise ValueError('Initial checkpoint did not contain requested gate')
    reports = {}
    for name, directory in (('control', control_root / 'control-depth'), ('gate', root / 'gate-depth')):
        summary_path, report_path = directory / 'summary.json', directory / 'depth-2.json'
        summary, report = json.loads(summary_path.read_text()), json.loads(report_path.read_text())
        identity = json.loads((directory / 'inputs.json').read_text())
        if (summary['status'] != 'complete' or summary['protocol'] != 'teacher'
                or not summary['same_serialized_bank'] or summary['writer_depth'] != 1
                or report['recurrence']['loops'] != 2
                or identity['episodes_sha256'] != lock['validation_sha256']):
            raise ValueError('Incomplete fixed-bank held-out evaluation')
        reports[name] = dict(summary=summary, report=report, identity=identity,
                             rows=_group(report), report_sha256=file_sha256(report_path),
                             summary_sha256=file_sha256(summary_path))
    control, gate = reports['control']['rows'], reports['gate']['rows']
    if set(control) != set(gate):
        raise ValueError('Held-out episode identities differ')
    for episode in control:
        for condition in CONDITIONS:
            left, right = control[episode][condition], gate[episode][condition]
            if (left['trajectory'], left['token_count'], left['selected_ids']) != (
                    right['trajectory'], right['token_count'], right['selected_ids']):
                raise ValueError('Target or selected source IDs differ')
    text = []
    for path, checkpoint_hash in ((control_root / 'control-text-r1' / 'results.json',
                                   reports['control']['identity']['checkpoint_model_sha256']),
                                  (root / 'gate-text-r1' / 'results.json',
                                   reports['gate']['identity']['checkpoint_model_sha256'])):
        report = json.loads(path.read_text())
        if (report['inputs']['episodes_sha256'] != lock['validation_sha256']
                or report['inputs']['checkpoint_model_sha256'] != checkpoint_hash
                or report['inputs']['depths'] != [1] or len(report['rows']) != 238):
            raise ValueError('Incomplete one-pass text control')
        text.append((report, file_sha256(path)))
    if text[0][0]['rows'] != text[1][0]['rows']:
        raise ValueError('One-pass fixed parent changed')
    result = dict(protocol='Same joint parent, 530 training episodes, seed and 400 Muon updates; initial memory gate only',
        analyzer_sha256=file_sha256(Path(__file__)), parent_model_sha256=lock['parent_model_sha256'],
        validation_sha256=lock['validation_sha256'], initial_memory_gate=initial_gate,
        prior_memory_gate=provenance['prior_memory_gate'],
        arms={name: dict(checkpoint_model_sha256=value['identity']['checkpoint_model_sha256'],
            report_sha256=value['report_sha256'], summary_sha256=value['summary_sha256'],
            writer_calls=value['summary']['write_phase']['writer_calls'],
            token_weighted_nll={condition: value['report']['summary'][condition]['token_weighted_nll']
                                for condition in CONDITIONS}) for name, value in reports.items()},
        one_pass_text=dict(report_sha256=[item[1] for item in text], all_238_scores_identical=True,
            selected_text_token_weighted_nll=text[0][0]['summary']['r1_selected_text']['token_weighted_nll']),
        paired={comparison: _paired(control, gate, comparison)
                for comparison in (*CONDITIONS, 'real_over_zero_values', 'real_over_wrong_values')},
        limits='Exploratory reused validation set; descriptive trajectory bootstrap, one training seed. A larger real-over-zero gap from zero-condition degradation alone is insufficient. Teacher NLL is not agent success, learned routing, composition, parameter substitution or cold-storage timing.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.control_root, args.output), indent=2))
