"""Answer-bearing episodes from the approved public retrieval corpora.

Replays the exact builders and settings that produced a published retrieval
corpus (`prepare_public_retrieval.py`), keeps queries whose dataset supplies a
short answer, and writes episodes in the Hotpot episode format. The answer is
only the supervised assistant target: queries never contain it. Positives are
the corpus's verified or answer-matched passages, and any one of them is a
sufficient group. Hard negatives go into the episode's supports as causally
available distractors, and all of them go into the bank source manifest.

Also writes the R5 source manifest: the Hotpot sources followed by every public
source that a selected episode references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

from prepare_public_retrieval import BUILDERS, _clean

PROMPT = 'Use the previously stored passages. Give only the short response.\nQuestion: '


def build(raw: Path, retrieval: Path, hotpot_sources: Path, output: Path, *,
          train_per_corpus: int, validation_per_corpus: int, max_answer_words: int,
          seed: int = 1701) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((retrieval / 'manifest.json').read_text())
    published = {}
    with (retrieval / 'sources.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            published[row['record_id']] = row
    output.mkdir(parents=True)
    counts, used = {}, {}
    for split, wanted in (('train', train_per_corpus), ('validation', validation_per_corpus)):
        episodes = []
        for name, builder in BUILDERS:
            limit = (manifest['limits'][name] if split == 'train'
                     else max(manifest['limits'][name] // 20, 200))
            kept = skipped = 0
            for query, positives, negatives, identifier, answer in builder(
                    raw, split, limit, manifest['seed']):
                answer = _clean(answer or '')
                if (not answer or len(answer.split()) > max_answer_words
                        or answer.lower() in query.lower()):
                    skipped += 1
                    continue
                positive_ids = list(dict.fromkeys(item['record_id'] for item in positives))
                supports = {item['record_id']: item for item in (*positives, *negatives)}
                if set(supports) - published.keys():
                    raise ValueError(f'{identifier} references an unpublished source')
                episodes.append({
                    'episode_id': identifier, 'environment': f'{name}-{split}',
                    'query': PROMPT + _clean(query), 'answer': answer, 'query_time': 2,
                    'required_ids': positive_ids,
                    'sufficient_groups': [[record_id] for record_id in positive_ids],
                    'support_annotation': ('verified' if name in {'msmarco', 'squad'}
                                           else 'answer_match'),
                    'task_family': 'public_qa',
                    'supports': [{'record_id': record_id, 'text': published[record_id]['text'],
                                  'created_at': 1, 'kind': 'passage'}
                                 for record_id in supports],
                    'provenance': {'dataset': name, 'domain': name, 'split': split},
                })
                used.update(dict.fromkeys(supports))
                kept += 1
                if kept == wanted:
                    break
            counts[f'{name}/{split}'] = {'episodes': kept, 'skipped_answers': skipped}
        random.Random(f'{seed}:{split}').shuffle(episodes)
        path = output / f'public-{split}.jsonl'
        with path.open('w', encoding='utf-8') as handle:
            for episode in episodes:
                handle.write(json.dumps(episode) + '\n')
    pending = output / 'sources-r5.jsonl.pending'
    hotpot_ids = set()
    with pending.open('w', encoding='utf-8') as handle:
        with hotpot_sources.open(encoding='utf-8') as source:
            for line in source:
                hotpot_ids.add(json.loads(line)['record_id'])
                handle.write(line if line.endswith('\n') else line + '\n')
        for record_id in sorted(used):
            if record_id in hotpot_ids:
                raise ValueError('Public and Hotpot source identities collide')
            handle.write(json.dumps(published[record_id]) + '\n')
    os.replace(pending, output / 'sources-r5.jsonl')
    result = {
        'format': 1, 'retrieval_manifest': manifest,
        'hotpot_sources': str(hotpot_sources), 'hotpot_source_count': len(hotpot_ids),
        'public_source_count': len(used), 'counts': counts,
        'train_per_corpus': train_per_corpus, 'validation_per_corpus': validation_per_corpus,
        'max_answer_words': max_answer_words, 'seed': seed,
        'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(output.glob('*.jsonl'))},
        'notice': ('Answers are supervised targets only, never query text. TriviaQA and '
                   'SearchQA positives are answer-matched snippets (distant supervision).'),
    }
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--hotpot-sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-per-corpus', type=int, default=5000)
    parser.add_argument('--validation-per-corpus', type=int, default=250)
    parser.add_argument('--max-answer-words', type=int, default=24)
    args = parser.parse_args()
    summary = build(args.raw, args.retrieval, args.hotpot_sources, args.output,
                    train_per_corpus=args.train_per_corpus,
                    validation_per_corpus=args.validation_per_corpus,
                    max_answer_words=args.max_answer_words)
    print(json.dumps({key: summary[key] for key in (
        'counts', 'hotpot_source_count', 'public_source_count')}, indent=2))
