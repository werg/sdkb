"""Verify native endpoint invariance of every frozen weight, payload and oracle row."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sqlite3

from safetensors import safe_open
import torch

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def verify(study, output):
    inputs = json.loads((study / 'inputs.json').read_text())
    source = resolve_checkpoint(Path(inputs['source_checkpoint']), verify=True)
    target = resolve_checkpoint(Path(inputs['configs']['routing_only']['run']), verify=True)
    prefixes = ('key_head.', 'address_maps.', 'query_maps.')
    changed, frozen = [], []
    torch.set_num_threads(2)
    with safe_open(source / 'model.safetensors', framework='pt') as a, safe_open(target / 'model.safetensors', framework='pt') as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError('State topology differs')
        for name in a.keys():
            same = torch.equal(a.get_tensor(name), b.get_tensor(name))
            if name.startswith(prefixes):
                if not same:
                    changed.append(name)
            else:
                if not same:
                    raise ValueError(f'Frozen weight changed: {name}')
                frozen.append(name)
    if not changed:
        raise ValueError('No address projection changed')
    root = study / 'confirmation'
    reports = [json.loads((root / name / 'results.json').read_text()) for name in ['source', 'routing_only']]
    for report, checkpoint in zip(reports, [source, target], strict=True):
        if (report['checkpoint_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json')
                or report['episodes_sha256'] != file_sha256(root / 'episodes.jsonl')):
            raise ValueError('Evaluation identity differs')
    controls = [{(r['episode'], r['condition']): r for r in report['rows']
                 if r['condition'] in {'selected_pair', 'none'}} for report in reports]
    if not controls[0] or controls[0] != controls[1]:
        raise ValueError('Oracle/no-memory rows differ')
    with ExitStack() as stack:
        connections = [sqlite3.connect(f'file:{root / name / "bank.sqlite"}?mode=ro', uri=True)
                       for name in ['source', 'routing_only']]
        for connection in connections:
            stack.callback(connection.close)
        sql = ('SELECT namespace,record_id,space,generation,domain,created_at,payload,source_id,deleted '
               'FROM records ORDER BY namespace,record_id,space,generation')
        payloads = [connection.execute(sql).fetchall() for connection in connections]
        if not payloads[0] or payloads[0] != payloads[1]:
            raise ValueError('Stored payload bytes or identities differ')
    record = {'source_manifest_sha256': file_sha256(source / 'manifest.json'),
              'target_manifest_sha256': file_sha256(target / 'manifest.json'),
              'episodes_sha256': file_sha256(root / 'episodes.jsonl'),
              'frozen_tensors_equal': len(frozen), 'changed_address_tensors': changed,
              'identical_control_rows': len(controls[0]), 'identical_payload_rows': len(payloads[0]),
              'notice': 'Exact invariance check, not evidence of learned routing capability.'}
    atomic_json(output, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify(args.study.resolve(), args.output.resolve())
