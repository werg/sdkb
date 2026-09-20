"""Seal both unused new repository splits as one fresh evaluation-only set."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def seal(root: Path, previous: Path):
    data = root / 'data'
    manifest = json.loads((data / 'manifest.json').read_text())
    lock = json.loads((root / 'inputs.json').read_text())['declaration']
    for name, digest in manifest['sha256'].items():
        if file_sha256(data / name) != digest:
            raise ValueError('New prepared data changed')
    if file_sha256(previous / 'data/normalized.jsonl') != lock['prior_normalized_sha256']:
        raise ValueError('Prior prepared data changed')
    if (manifest['sources'][0]['spec'] != lock['source']
            or manifest['sources'][0]['counts']['rows_skipped'] != lock['skip_rows']
            or manifest['sources'][0]['counts']['rows_scanned'] != lock['scan_rows']):
        raise ValueError('New preparation differs from pinned stream declaration')
    prior_rows = [json.loads(line) for line in
                  (previous / 'data/normalized.jsonl').read_text().splitlines()]
    prior_groups = {row['split_group'] for row in prior_rows}
    prior_trajectories = {row['trajectory_id'] for row in prior_rows}
    if len(prior_groups) != lock['excluded_groups']:
        raise ValueError('Prior group count changed')
    rows = []
    split_groups = {}
    for split in ('train', 'validation'):
        subset = [json.loads(line) for line in (data / f'{split}.jsonl').read_text().splitlines()]
        if len(subset) != manifest['summary'][split]['episodes']:
            raise ValueError('Incomplete episode subset')
        split_groups[split] = {row['provenance']['split_group'] for row in subset}
        rows.extend(subset)
    groups = split_groups['train'] | split_groups['validation']
    if (not rows or groups & prior_groups or split_groups['train'] & split_groups['validation']
            or {row['provenance']['trajectory_id'] for row in rows} & prior_trajectories
            or len({row['episode_id'] for row in rows}) != len(rows)
            or any(not row['provenance']['target_complete'] or not row['answer']
                   or any(s['created_at'] >= row['query_time'] for s in row['supports']) for row in rows)):
        raise ValueError('New episodes violate freshness, identity or causal checks')
    target = root / 'evaluation.jsonl'
    pending = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with pending.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        digest = file_sha256(pending)
        seal_file = root / 'evaluation-seal.json'
        summary = {'protocol': 'Both new, unused subsets reserved for evaluation only',
                   'source_revision': lock['source_revision'],
                   'previous_normalized_sha256': lock['prior_normalized_sha256'],
                   'preparation_manifest_sha256': file_sha256(data / 'manifest.json'),
                   'train_sha256': manifest['sha256']['train.jsonl'],
                   'validation_sha256': manifest['sha256']['validation.jsonl'],
                   'evaluation_sha256': digest, 'episodes': len(rows),
                   'trajectories': len({row['provenance']['trajectory_id'] for row in rows}),
                   'repository_groups': len(groups), 'prior_repository_groups': len(prior_groups),
                   'group_overlap': 0, 'source_trajectories_overlap': 0,
                   'target_tokens': manifest['summary']['train']['complete_target_tokens']
                                    + manifest['summary']['validation']['complete_target_tokens'],
                   'preparation_commit': '64db124',
                   'notice': 'Data-only freshness seal; no model has been scored on these episodes. Later comparisons need a predeclared objective and matched controls.'}
        if target.exists() or seal_file.exists():
            if (not target.exists() or not seal_file.exists()
                    or json.loads(seal_file.read_text()) != summary
                    or file_sha256(target) != digest):
                raise ValueError('Existing sealed evaluation differs')
        else:
            os.replace(pending, target)
            atomic_json(seal_file, summary)
        return summary
    finally:
        pending.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.root, args.previous), indent=2))
