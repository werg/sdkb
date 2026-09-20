"""Build a bounded reconstruction/QA pilot from the external SQuAD source manifest."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import sdkb.corpus_data as corpus_data
from sdkb.corpus_data import short_reconstruction
from sdkb.data import Source
from sdkb.trajectories import file_sha256

REVISION = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'


def prepare(data: Path, output: Path, tokenizer, *, source_count: int = 1024,
            validation_count: int = 64, max_words: int = 40,
            max_target_tokens: int = 128) -> dict:
    if min(source_count, validation_count, max_words, max_target_tokens) < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive budgets and a fresh output under an existing parent required')
    if output.parent.stat().st_dev == Path(__file__).resolve().parents[1].stat().st_dev:
        raise ValueError('Pilot data must live on the external disk')
    if shutil.disk_usage(output.parent).free < 10 * 1024**3:
        raise OSError('Pilot preparation requires a 10 GiB disk reserve')
    manifest = json.loads((data / 'manifest.json').read_text())
    for name in ('sources-train.jsonl', 'train.jsonl',
                 'sources-validation.jsonl', 'validation.jsonl'):
        if file_sha256(data / name) != manifest['sha256'][name]:
            raise ValueError(f'Prepared source changed: {name}')
    pending = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
    pending.mkdir()
    try:
        summaries = {}
        for split, count in (('train', source_count), ('validation', validation_count)):
            source_rows = [json.loads(line) for line in
                           (data / f'sources-{split}.jsonl').read_text().splitlines()]
            if len(source_rows) < count:
                raise ValueError('Source budget exceeds prepared corpus')
            selected = {row['record_id']: Source(**{key: row[key] for key in
                                                   ('record_id', 'text', 'created_at', 'kind')})
                        for row in source_rows[:count]}
            qa = {}
            for line in (data / f'{split}.jsonl').open():
                row = json.loads(line)
                source_id = row['required_ids'][0]
                if source_id in selected and source_id not in qa:
                    qa[source_id] = row
            if len(qa) != len(selected):
                raise ValueError('Every selected source needs a verified QA')
            episodes = []
            for source_id, source in selected.items():
                episodes.extend((asdict(short_reconstruction(
                    source, max_words=max_words, tokenizer=tokenizer,
                    max_target_tokens=max_target_tokens)), qa[source_id]))
            for episode in episodes:
                tokens = len(tokenizer.encode(episode['answer'], add_special_tokens=False))
                if tokens + int(tokenizer.eos_token_id is not None) > max_target_tokens:
                    raise ValueError('Complete target exceeds the declared token budget')
            episodes.sort(key=lambda row: hashlib.sha256(row['episode_id'].encode()).digest())
            path = pending / f'{split}.jsonl'
            with path.open('w') as handle:
                for episode in episodes:
                    handle.write(json.dumps(episode, sort_keys=True, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            summaries[split] = {'sources': count, 'episodes': len(episodes),
                                'sha256': file_sha256(path)}
        result = {'format': 2, 'source_manifest_sha256': file_sha256(data / 'sources-train.jsonl'),
                  'prepared_manifest_sha256': file_sha256(data / 'manifest.json'),
                  'preparer_sha256': file_sha256(Path(__file__)),
                  'corpus_module_sha256': file_sha256(Path(corpus_data.__file__)),
                  'source_count': source_count, 'max_words': max_words,
                  'max_target_tokens': max_target_tokens,
                  'tokenizer_revision': REVISION,
                  'splits': summaries,
                  'notice': 'One source record per task, supplied oracle selection; this pilot does not train global retrieval.'}
        (pending / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
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
    parser.add_argument('--source-count', type=int, default=1024)
    parser.add_argument('--validation-count', type=int, default=64)
    parser.add_argument('--max-words', type=int, default=40)
    parser.add_argument('--max-target-tokens', type=int, default=128)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    snapshot = args.cache / 'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots' / REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    print(json.dumps(prepare(args.data, args.output, tokenizer,
                             source_count=args.source_count,
                             validation_count=args.validation_count,
                             max_words=args.max_words,
                             max_target_tokens=args.max_target_tokens), indent=2))
