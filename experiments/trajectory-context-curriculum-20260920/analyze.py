"""Compare the source-utility curriculum with its uniform-data Muon control."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random

from sdkb.config import load_config
from sdkb.trajectories import file_sha256


CONDITIONS = ('all', 'zero_values', 'wrong_values', 'none')


def _rows(report):
    grouped = defaultdict(dict)
    for row in report['rows']:
        if row['condition'] in grouped[row['episode']]:
            raise ValueError('Duplicate validation condition')
        grouped[row['episode']][row['condition']] = row
    if len(grouped) != 119 or any(set(value) != set(CONDITIONS) for value in grouped.values()):
        raise ValueError('Expected 119 complete held-out episodes')
    for value in grouped.values():
        if (value['all']['selected_ids'] != value['zero_values']['selected_ids']
                or value['all']['selected_ids'] != value['wrong_values']['selected_ids']):
            raise ValueError('Payload intervention changed selected IDs')
    return grouped


def _paired_delta(control, selected, fn):
    values = defaultdict(list)
    for episode in control:
        values[control[episode]['all']['trajectory']].append(fn(control[episode], selected[episode]))
    groups = list(values.values())
    rng = random.Random(233)
    samples = []
    for _ in range(2000):
        sample = [rng.choice(groups) for _ in groups]
        samples.append(sum(map(sum, sample)) / sum(map(len, sample)))
    samples.sort()
    return dict(mean_episode_delta=sum(map(sum, groups)) / sum(map(len, groups)),
                trajectory_groups=len(groups), trajectory_bootstrap_interval_95=[samples[50], samples[1949]])


def analyze(root: Path, control_root: Path, output: Path) -> dict:
    lock = json.loads((root / 'inputs.json').read_text())
    control_lock = json.loads((control_root / 'inputs.json').read_text())
    if (lock['parent_model_sha256'] != control_lock['parent_model_sha256']
            or lock['validation_sha256'] != control_lock['validation_sha256']):
        raise ValueError('Parent or validation set differs from matched control')
    control_config = asdict(load_config(control_root / 'control.yaml'))
    selected_config = asdict(load_config(root / 'selected-context.yaml'))
    if (selected_config['train']['episodes_file'] != lock['selected_train']
            or file_sha256(root / 'selected-context.yaml') != lock['config_sha256']):
        raise ValueError('Curriculum data or config changed')
    for config in (control_config, selected_config):
        config['train'].pop('episodes_file')
        config['train'].pop('wandb_group')
    if control_config != selected_config:
        raise ValueError('Non-curriculum model or optimizer settings differ')
    reports = {}
    for arm, directory in (('uniform', control_root / 'control-depth'),
                           ('selected', root / 'selected-depth')):
        summary_path = directory / 'summary.json'
        report_path = directory / 'depth-2.json'
        summary = json.loads(summary_path.read_text())
        report = json.loads(report_path.read_text())
        identity = json.loads((directory / 'inputs.json').read_text())
        if (summary['status'] != 'complete' or summary['protocol'] != 'teacher'
                or not summary['same_serialized_bank'] or summary['writer_depth'] != 1
                or report['recurrence']['loops'] != 2
                or identity['episodes_sha256'] != lock['validation_sha256']):
            raise ValueError('Invalid fixed-bank held-out evaluation')
        reports[arm] = dict(summary=summary, report=report, identity=identity,
                            rows=_rows(report), report_sha256=file_sha256(report_path),
                            summary_sha256=file_sha256(summary_path))
    uniform, selected = reports['uniform']['rows'], reports['selected']['rows']
    if set(uniform) != set(selected):
        raise ValueError('Validation episode IDs differ')
    for episode in uniform:
        for condition in CONDITIONS:
            a, b = uniform[episode][condition], selected[episode][condition]
            if (a['trajectory'], a['token_count'], a['selected_ids']) != (
                    b['trajectory'], b['token_count'], b['selected_ids']):
                raise ValueError('Matched episode/source identity changed')
    text_reports = []
    for path, checkpoint_hash in ((control_root / 'control-text-r1' / 'results.json',
                                   reports['uniform']['identity']['checkpoint_model_sha256']),
                                  (root / 'selected-text-r1' / 'results.json',
                                   reports['selected']['identity']['checkpoint_model_sha256'])):
        text = json.loads(path.read_text())
        if (text['inputs']['episodes_sha256'] != lock['validation_sha256']
                or text['inputs']['checkpoint_model_sha256'] != checkpoint_hash
                or text['inputs']['depths'] != [1] or len(text['rows']) != 238):
            raise ValueError('Incomplete or mismatched one-pass text control')
        text_reports.append((text, file_sha256(path)))
    if text_reports[0][0]['rows'] != text_reports[1][0]['rows']:
        raise ValueError('Frozen one-pass text teacher changed')
    paired = {f'uniform_to_selected_{condition}_nll_reduction': _paired_delta(
        uniform, selected, lambda a, b, c=condition: a[c]['mean_nll'] - b[c]['mean_nll'])
        for condition in CONDITIONS}
    for other in ('zero_values', 'wrong_values'):
        paired[f'curriculum_delta_real_over_{other}'] = _paired_delta(uniform, selected,
            lambda a, b, c=other: ((b[c]['mean_nll'] - b['all']['mean_nll'])
                                    - (a[c]['mean_nll'] - a['all']['mean_nll'])))
    result = dict(protocol='Same joint parent and 400 Muon updates; training upper-half text-utility selection vs uniform data',
        analyzer_sha256=file_sha256(Path(__file__)), parent_model_sha256=lock['parent_model_sha256'],
        validation_sha256=lock['validation_sha256'], selected_train_sha256=lock['selected_train_sha256'],
        training_selection={key: lock[key] for key in ('selected_episodes', 'omitted_episodes',
            'selected_trajectories', 'omitted_trajectories', 'mean_text_gain_selected', 'mean_text_gain_omitted')},
        arms={arm: dict(checkpoint_model_sha256=value['identity']['checkpoint_model_sha256'],
            report_sha256=value['report_sha256'], summary_sha256=value['summary_sha256'],
            writer_calls=value['summary']['write_phase']['writer_calls'],
            token_weighted_nll={condition: value['report']['summary'][condition]['token_weighted_nll']
                                for condition in CONDITIONS}) for arm, value in reports.items()},
        one_pass_text=dict(report_sha256=[value[1] for value in text_reports],
            selected_text_token_weighted_nll=text_reports[0][0]['summary']['r1_selected_text']['token_weighted_nll'],
            no_text_token_weighted_nll=text_reports[0][0]['summary']['r1_none']['token_weighted_nll'],
            all_238_scores_identical=True), paired=paired,
        limits='Teacher-forced likelihood and descriptive trajectory bootstrap, not task success. Training example distributions differ by design, and both arms share one parent and seed. Oracle selection and fixed plans do not establish learned routing; text and latent paths use different token and compute budgets. No capacity-substitution or cold-storage timing claim.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--control-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.control_root, args.output), indent=2))
