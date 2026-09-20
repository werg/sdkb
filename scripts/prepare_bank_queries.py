"""Select source-held-out SQuAD questions for an already published train bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from sdkb.data import episode_from_dict
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256

REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def prepare(data: Path, output: Path, tokenizer, *, source_offset: int = 1024,
            source_count: int = 64, max_target_tokens: int = 128) -> dict:
    if (source_offset < 0 or min(source_count, max_target_tokens) < 1 or output.exists()
            or not output.parent.is_dir()):
        raise ValueError('Positive budgets and fresh output under an existing parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Evaluation data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Evaluation preparation requires a 10 GiB disk reserve')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('sources-train.jsonl', 'train.jsonl'):
        if file_sha256(data / name) != manifest['sha256'][name]:
            raise ValueError(f'Prepared source changed: {name}')
    source_rows = [json.loads(line) for line in (data / 'sources-train.jsonl').open()]
    if source_offset + source_count > len(source_rows):
        raise ValueError('Held-out source range exceeds published source manifest')
    ids = [row['record_id'] for row in source_rows[source_offset:source_offset + source_count]]
    chosen = {}
    for line in (data / 'train.jsonl').open():
        row = json.loads(line)
        source_id = row['required_ids'][0]
        if source_id in ids and source_id not in chosen:
            episode = episode_from_dict(row)
            if (len(tokenizer.encode(episode.answer, add_special_tokens=False))
                    + int(tokenizer.eos_token_id is not None) <= max_target_tokens):
                chosen[source_id] = row
    if len(chosen) != len(ids):
        raise ValueError('Every held-out source needs a bounded verified question')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        path = pending / 'episodes.jsonl'
        with path.open('w') as handle:
            for source_id in ids:
                handle.write(json.dumps(chosen[source_id], sort_keys=True, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        result = {'format': 1, 'prepared_manifest_sha256': file_sha256(data / 'manifest.json'),
                  'source_manifest_sha256': file_sha256(data / 'sources-train.jsonl'),
                  'source_offset': source_offset, 'source_count': source_count,
                  'source_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                  'episodes_sha256': file_sha256(path), 'max_target_tokens': max_target_tokens,
                  'tokenizer_revision': REVISION,
                  'notice': 'Questions and sources were excluded from the first 1024-source reconstruction stage; '
                            'all selected source IDs belong to the later 10k bank.'}
        atomic_json(pending / 'manifest.json', result)
        os.replace(pending, output)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-offset', type=int, default=1024)
    parser.add_argument('--source-count', type=int, default=64)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    snapshot = args.cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.data, args.output, tokenizer,
                             source_offset=args.source_offset, source_count=args.source_count), indent=2))
