"""Confirm frozen-feature address endpoints through actual stored-memory behavior."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate_binding_context import evaluate as choices
from evaluate_stored_generation import evaluate as generate
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import make_multiuse_world, save_episodes
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.trajectories import file_sha256


def run(source, probes, output):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False):
        checkpoint = resolve_checkpoint(source, verify=True)
        episodes = output / 'episodes.jsonl'
        if not episodes.exists():
            save_episodes(episodes, [e for i in range(32) for e in make_multiuse_world(
                i, split='routing-stored-confirmation-20260919', bindings=2)])
        arms = {'source': (None, False), 'compressed_query': (probes / 'compressed_query-resume.pt', False),
                'full_state_query': (probes / 'full_state_query-resume.pt', True)}
        identity = {'source_checkpoint': str(checkpoint), 'source_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
                    'episodes_sha256': file_sha256(episodes),
                    'probes': {name: file_sha256(path) if path else None for name, (path, _) in arms.items()}}
        if (output / 'inputs.json').exists() and json.loads((output / 'inputs.json').read_text()) != identity:
            raise ValueError('Evaluation identity changed')
        atomic_json(output / 'inputs.json', identity)
        for name, (probe, independent) in arms.items():
            if stop_requested(output):
                raise RuntimeError('Evaluation stopped')
            destination = output / name
            if not (destination / 'results.json').exists():
                choices(checkpoint, episodes, destination, learned_world=True, read_budget=2,
                        routing_probe=probe, independent_routing_query=independent)
            report = json.loads((destination / 'results.json').read_text())
            if report['episodes_sha256'] != identity['episodes_sha256']:
                raise ValueError('Episode identity differs')
            generation_path = output / f'{name}-generation.json'
            if not generation_path.exists():
                result = generate(checkpoint, destination / 'bank.sqlite', episodes, 8, 24,
                                  learned_world=True, read_budget=2, routing_probe=probe,
                                  independent_routing_query=independent)
                atomic_json(generation_path, result)
            print(json.dumps({'completed': name}), flush=True)
        reports = {name: json.loads((output / name / 'results.json').read_text()) for name in arms}
        controls = {name: [r for r in report['rows'] if r['condition'] in {'selected_pair', 'none'}]
                    for name, report in reports.items()}
        if any(rows != controls['source'] for rows in controls.values()):
            raise ValueError('Frozen oracle/no-memory behavior changed')
        sql = ('SELECT namespace,record_id,space,generation,domain,created_at,payload,source_id,deleted '
               'FROM records ORDER BY namespace,record_id,space,generation')
        payloads = {}
        for name in arms:
            connection = sqlite3.connect(f'file:{output / name / "bank.sqlite"}?mode=ro', uri=True)
            try:
                payloads[name] = connection.execute(sql).fetchall()
            finally:
                connection.close()
        if any(values != payloads['source'] for values in payloads.values()):
            raise ValueError('Frozen payload bytes or provenance changed')
        atomic_json(output / 'frozen-path.json', {'identical_control_rows_per_arm': len(controls['source']),
                    'identical_payload_rows_per_arm': len(payloads['source']), 'arms': list(arms), 'identity': identity})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--probes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.probes, args.output)
