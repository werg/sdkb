"""Prepare approved public retrieval corpora for keyspace pretraining.

Each corpus becomes its own domain namespace. Sources are stored passages;
queries name their verified positive sources and, where the corpus provides
them, hard negatives. Answers are never placed in queries. Every source is
created before every query (``created_at`` 1, ``query_time`` 2), so causal
eligibility is the whole namespace.
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
from pathlib import Path
import random
import re

import pyarrow.parquet as pq

MAX_CHARS = 700


def _clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()[:MAX_CHARS]


def _source(corpus: str, text: str, title: str = '') -> dict:
    body = f'Title: {_clean(title)}\nPassage: {_clean(text)}' if title else _clean(text)
    record_id = hashlib.sha256(f'{corpus}\0{body}'.encode()).hexdigest()[:32]
    return {'record_id': record_id, 'text': body, 'domain': corpus, 'created_at': 1,
            'kind': 'passage', 'provenance': {'dataset': corpus}}


def _rows(pattern: str, limit: int, seed: int):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    rows = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def _literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def msmarco(raw: Path, split: str, limit: int, seed: int):
    for row in _rows(str(raw / f'datasets--ms_marco/v2.1/{split}-*.parquet'), limit, seed):
        passages = _literal(row['passages'])
        positives, negatives = [], []
        for text, selected in zip(passages['passage_text'], passages['is_selected'],
                                  strict=True):
            (positives if selected else negatives).append(_source('msmarco', text))
        if positives:
            yield row['query'], positives, negatives, f"msmarco-{row['query_id']}"


def squad(raw: Path, split: str, limit: int, seed: int):
    for row in _rows(str(raw / f'datasets--rajpurkar--squad/plain_text/{split}-*.parquet'),
                     limit, seed):
        title = row['title'].replace('_', ' ')
        yield row['question'], [_source('squad', row['context'], title)], [], f"squad-{row['id']}"


def _contains(text: str, aliases: list[str]) -> bool:
    lowered = text.lower()
    return any(alias and alias.lower() in lowered for alias in aliases)


def triviaqa(raw: Path, split: str, limit: int, seed: int):
    for row in _rows(str(raw / f'datasets--mandarjoshi--trivia_qa/rc/{split}-*.parquet'),
                     limit, seed):
        answer = _literal(row['answer'])
        aliases = [answer['value'], *answer['aliases']]
        results = _literal(row['search_results'])
        positives, negatives = [], []
        for title, text in zip(results['title'], results['description'], strict=True):
            if text:
                (positives if _contains(text, aliases) else negatives).append(
                    _source('triviaqa', text, title))
        if positives:
            yield row['question'], positives[:3], negatives[:5], f"triviaqa-{row['question_id']}"


def searchqa(raw: Path, split: str, limit: int, seed: int):
    for row in _rows(str(raw / f'datasets--lucadiliello--searchqa/data/{split}-*.parquet'),
                     limit, seed):
        answers = _literal(row['answers'])
        positives, negatives = [], []
        for document in row['context'].split('[DOC]'):
            match = re.match(r'\s*\[TLE\](.*?)\[PAR\](.*)', document, re.S)
            if not match or not match.group(2).strip():
                continue
            source = _source('searchqa', match.group(2), match.group(1))
            (positives if _contains(match.group(2), answers) else negatives).append(source)
        if positives:
            yield row['question'], positives[:3], negatives[:5], f"searchqa-{row['key']}"


def prepare(raw: Path, output: Path, *, limits: dict[str, int], seed: int = 1701) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sources: dict[str, dict] = {}
    counts = {}
    with (output / 'queries.jsonl.pending').open('w', encoding='utf-8') as handle:
        for name, builder in (('msmarco', msmarco), ('squad', squad),
                              ('triviaqa', triviaqa), ('searchqa', searchqa)):
            for split, suffix in (('train', 'train'), ('validation', 'validation')):
                limit = limits[name] if split == 'train' else max(limits[name] // 20, 200)
                written = 0
                for query, positives, negatives, identifier in builder(raw, split, limit, seed):
                    for item in (*positives, *negatives):
                        sources.setdefault(item['record_id'], item)
                    positive_ids = list(dict.fromkeys(item['record_id'] for item in positives))
                    negative_ids = [record_id for record_id in dict.fromkeys(
                        item['record_id'] for item in negatives) if record_id not in positive_ids]
                    handle.write(json.dumps({
                        'episode_id': identifier, 'query': _clean(query), 'domain': name,
                        'query_time': 2, 'required_ids': positive_ids,
                        'hard_negative_ids': negative_ids, 'split': suffix,
                        'support_annotation': ('verified' if name in {'msmarco', 'squad'}
                                               else 'answer_match')}) + '\n')
                    written += 1
                counts[f'{name}/{split}'] = written
    os.replace(output / 'queries.jsonl.pending', output / 'queries.jsonl')
    with (output / 'sources.jsonl').open('w', encoding='utf-8') as handle:
        for record_id in sorted(sources):
            handle.write(json.dumps(sources[record_id]) + '\n')
    per_domain = {}
    for row in sources.values():
        per_domain[row['domain']] = per_domain.get(row['domain'], 0) + 1
    manifest = {
        'format': 1, 'raw': str(raw), 'limits': limits, 'seed': seed,
        'revisions': (raw / 'REVISIONS.txt').read_text().splitlines(),
        'queries': counts, 'sources': len(sources), 'sources_per_domain': per_domain,
        'license_note': 'Owner-approved for keyspace pretraining, 23 September 2026.',
        'notice': ('Answers are never in queries. Positives for TriviaQA and SearchQA are '
                   'answer-matched snippets (distant supervision), not human labels.'),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--msmarco', type=int, default=40000)
    parser.add_argument('--squad', type=int, default=60000)
    parser.add_argument('--triviaqa', type=int, default=30000)
    parser.add_argument('--searchqa', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.output, limits={
        'msmarco': args.msmarco, 'squad': args.squad, 'triviaqa': args.triviaqa,
        'searchqa': args.searchqa}, seed=args.seed), indent=2))
