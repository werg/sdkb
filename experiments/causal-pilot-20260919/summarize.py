"""Extract aggregate evidence; run inside the training environment after evaluation."""
from pathlib import Path
import argparse
import json
import math

from sdkb.metrics import paired_world_bootstrap, summarize_rows


def summarize(run):
    comparison = json.loads((run / 'stage-transfer-comparison/comparison.json').read_text())
    report = {'scope': 'One training seed; controlled two-bit transfer, not agent success or capacity substitution.',
              'stages': {}}
    for name, stage in comparison['stages'].items():
        original = stage['report']
        rows = json.loads((Path(original['directory']) / 'results.json').read_text())['rows']
        by_family = {}
        for family in sorted({r['task_family'] for r in rows}):
            subset = [r for r in rows if r['task_family'] == family]
            by_family[family] = {
                'conditions': summarize_rows(subset),
                'paired_gains': {condition: paired_world_bootstrap(subset, b=condition)
                                 for condition in ('none', 'zero_values', 'drop_0', 'drop_1')}}
        training = json.loads((run / name / 'training_summary.json').read_text())
        metrics = [json.loads(line) for line in (run / name / 'metrics.jsonl').read_text().splitlines()]
        if training['stopped_early']:
            raise ValueError(f'Incomplete stage: {name}')
        nonfinite = [row['step'] for row in metrics if any(
            isinstance(value, float) and not math.isfinite(value) for value in row.values())]
        report['stages'][name] = {
            'identity': stage['identity'], 'by_family': by_family,
            'counterfactuals': original['counterfactuals'],
            'xor_counterfactuals': original['xor_counterfactuals'],
            'steps': training['steps'], 'last_training_elapsed_seconds': metrics[-1]['elapsed_seconds'],
            'nonfinite_metric_steps': nonfinite, 'training_resources': training['resources'],
            'environment': training['environment'], 'write_phase': original['write_phase']}
    report['out_of_distribution_binding'] = json.loads((run / 'multiuse-evaluation.json').read_text())
    binding = report['out_of_distribution_binding']
    rows = json.loads((Path(binding['directory']) / 'results.json').read_text())['rows']
    report['binding_paired_gains'] = {
        family: {condition: paired_world_bootstrap(
            [r for r in rows if r['task_family'] == family], b=condition)
                 for condition in ('none', 'zero_values')}
        for family in sorted({r['task_family'] for r in rows})}
    report['depth_sweep'] = json.loads((run / 'depth-sweep/summary.json').read_text())
    report['depth_dataset'] = '32 validation worlds, 96 questions; separate from the final post-freeze comparison worlds.'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.run), indent=2) + '\n')
