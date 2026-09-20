"""Prepare a new repository-disjoint slice of the pinned public trajectory stream."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile


SKIP_ROWS = 512
SCAN_ROWS = 2048


def prepare(root: Path, cache: Path, previous: Path):
    if not root.parent.is_dir() or not cache.is_dir() or not previous.is_dir():
        raise FileNotFoundError('External root parent, cache and prior data must exist')
    if root.parent.stat().st_dev == Path(__file__).resolve().parents[2].stat().st_dev:
        raise ValueError('The prepared dataset must live on the external disk')
    if shutil.disk_usage(root.parent).free < 10 * 1024**3:
        raise OSError('Preparation requires a 10 GiB disk reserve')
    for name, relative in [('HF_HOME', 'huggingface'), ('HF_HUB_CACHE', 'huggingface/hub'),
                           ('HF_DATASETS_CACHE', 'huggingface/datasets'), ('HF_XET_CACHE', 'huggingface/xet'),
                           ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp/trajectory-fresh-holdout')]:
        path = cache / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    tempfile.tempdir = os.environ['TMPDIR']

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Preparation stopped; pinned input lock remains')
    signal.signal(signal.SIGTERM, interrupted)

    from transformers import AutoTokenizer
    from sdkb.operations import atomic_json
    from sdkb.trajectories import prepare_trajectories, file_sha256

    prior_lock = json.loads((previous / 'inputs.json').read_text())
    prior_manifest = json.loads((previous / 'data/manifest.json').read_text())
    for name, digest in prior_manifest['sha256'].items():
        if file_sha256(previous / 'data' / name) != digest:
            raise ValueError('Prior prepared data changed')
    old_source = prior_lock['pinned_source']
    if (old_source['path'] != 'SWE-bench/SWE-smith-trajectories'
            or old_source['max_rows'] != SKIP_ROWS
            or old_source['shuffle_buffer'] != 256 or old_source['shuffle_seed'] != 191):
        raise ValueError('Expected the pinned initial 512-row trajectory sample')
    prior_groups = sorted({json.loads(line)['split_group'] for line in
                           (previous / 'data/normalized.jsonl').read_text().splitlines()})
    if not prior_groups:
        raise ValueError('Prior sample has no repository groups')
    source = old_source | {'skip_rows': SKIP_ROWS, 'max_rows': SCAN_ROWS,
                           'excluded_split_groups': prior_groups}
    model_revision = prior_lock['declaration']['tokenizer_revision']
    snapshot = cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / model_revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer_hashes = {p.name: file_sha256(p) for p in snapshot.iterdir()
                        if p.name in {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'}}
    if tokenizer_hashes != prior_lock['declaration']['tokenizer_files']:
        raise ValueError('Tokenizer bytes differ from original preparation')
    declaration = {'protocol': 'prefix', 'source': source,
                   'source_revision': old_source['revision'],
                   'skip_rows': SKIP_ROWS, 'scan_rows': SCAN_ROWS,
                   'excluded_groups': len(prior_groups),
                   'prior_inputs_sha256': file_sha256(previous / 'inputs.json'),
                   'prior_manifest_sha256': file_sha256(previous / 'data/manifest.json'),
                   'prior_normalized_sha256': file_sha256(previous / 'data/normalized.jsonl'),
                   'budgets': prior_lock['declaration']['budgets'],
                   'tokenizer_revision': model_revision,
                   'tokenizer_files': tokenizer_hashes,
                   'seed': 191, 'validation_fraction': .2,
                   'notice': 'Public recorded data only; later rows in the same pinned shuffled stream. No paid teacher or tool execution. Row count is not a download-byte limit.'}
    root.mkdir(exist_ok=True)
    lock = root / 'inputs.json'
    if lock.exists():
        if json.loads(lock.read_text())['declaration'] != declaration:
            raise ValueError('Preparation inputs changed; choose a fresh root')
    else:
        atomic_json(lock, {'declaration': declaration, 'preparer_sha256': file_sha256(__file__)})
    data = root / 'data'
    if data.exists():
        manifest = json.loads((data / 'manifest.json').read_text())
        if any(file_sha256(data / name) != digest for name, digest in manifest['sha256'].items()):
            raise ValueError('Prepared data changed')
        return manifest
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=root) as tmp:
        pending = Path(tmp) / 'data'
        manifest = prepare_trajectories([source], tokenizer, declaration['budgets'], pending,
                                        protocol='prefix', seed=191, validation_fraction=.2)
        new_groups = {json.loads(line)['split_group'] for line in
                      (pending / 'normalized.jsonl').read_text().splitlines()}
        if new_groups & set(prior_groups):
            raise ValueError('New preparation overlaps prior repository groups')
        os.replace(pending, data)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.cache, args.previous), indent=2), flush=True)
