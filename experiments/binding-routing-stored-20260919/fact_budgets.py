"""Test single-fact interference with fixed one-versus-two-record read budgets."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate_stored_generation import evaluate
from sdkb.data import load_episodes, save_episodes
from sdkb.metrics import paired_world_bootstrap
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.trajectories import file_sha256


def compare(study, probes):
    inputs = json.loads((study / 'inputs.json').read_text())
    bank = study / 'full_state_query' / 'bank.sqlite'
    if not (bank.parent / 'results.json').exists():
        raise RuntimeError('Wait for the full-state choice evaluation to commit')
    episodes = [e for e in load_episodes(study / 'episodes.jsonl')
                if e.task_family in {'multiuse/permission', 'multiuse/restoration'}]
    if any(len(e.required_ids) != 1 for e in episodes):
        raise ValueError('This diagnostic requires single-fact queries')
    output = study / 'fact-budgets'
    output.mkdir(exist_ok=True)
    with run_lock(output, clear_stop=False):
        path = output / 'episodes.jsonl'
        if path.exists() and load_episodes(path) != episodes:
            raise ValueError('Diagnostic data changed')
        if not path.exists():
            save_episodes(path, episodes)
        reports = {}
        for budget in (1, 2):
            if stop_requested(output):
                raise RuntimeError('Fact-budget diagnostic stopped')
            destination = output / f'budget-{budget}.json'
            if destination.exists():
                raise FileExistsError(destination)
            report = evaluate(Path(inputs['source_checkpoint']), bank, path, 32, 24,
                              learned_world=True, read_budget=budget,
                              routing_probe=probes / 'full_state_query-resume.pt', independent_routing_query=True)
            atomic_json(destination, report)
            reports[budget] = report
        paired = {}
        for family in ('multiuse/permission', 'multiuse/restoration'):
            paired[family] = {}
            for condition in ('all', 'zero_values', 'none'):
                rows = [dict(row, choice_correct=row['exact_match'], condition=str(budget))
                        for budget, report in reports.items() for row in report['rows']
                        if row['task_family'] == family and row['condition'] == condition]
                paired[family][condition] = paired_world_bootstrap(rows, '1', '2')
        atomic_json(output / 'summary.json', {'episodes_sha256': file_sha256(path),
                    'by_budget': {str(k): v['by_family'] for k, v in reports.items()},
                    'one_minus_two': paired,
                    'notice': 'Single-fact subset and fixed budgets; not an adaptive learned read-count policy. No action questions scored.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--probes', type=Path, required=True)
    args = parser.parse_args()
    compare(args.study, args.probes)
