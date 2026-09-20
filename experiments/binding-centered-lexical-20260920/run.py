"""Center fixed lexical keys using only independent prior training content."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import runpy

import numpy as np

from sdkb.archiving import ensure_free
from sdkb.checkpoints import _atomic_text, stop_on_signal
from sdkb.data import load_episodes
from sdkb.episode_index import EpisodeIndex
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.trajectories import file_sha256


DENSE_PATH = Path(__file__).parents[2]/'scripts/evaluate_dense_lexical_routing.py'
DENSE = runpy.run_path(str(DENSE_PATH))


def normalize(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def calibrate(episodes, *, width, seed):
    sources = {}
    for episode in episodes:
        for source in episode.supports:
            if source.created_at >= episode.query_time:
                raise ValueError('Future source in calibration episode')
            if source.record_id in sources and sources[source.record_id] != source:
                raise ValueError('Changed calibration source identity')
            sources[source.record_id] = source
    def mean(texts):
        encoded = [DENSE['encode'](DENSE['LEXICAL']['terms'](text), width=width, seed=seed) for text in texts]
        return np.stack(encoded).mean(axis=0, dtype=np.float64).astype(np.float32)
    return np.stack([mean(sorted(s.text for s in sources.values())), mean(e.query for e in episodes)])


def publish(output, records, keys, centers, identity):
    output.mkdir(parents=True, exist_ok=True)
    centered = normalize(keys-centers[0])
    if (output/'manifest.json').exists():
        old_records, old_keys, old_centers = reopen(output, identity)
        if old_records != records or not np.array_equal(old_keys, centered) or not np.array_equal(old_centers, centers):
            raise ValueError('Existing centered index differs')
        return
    temporary = output/'vectors.npz.pending'
    try:
        with temporary.open('wb') as file:
            np.savez(file, keys=centered, centers=centers)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(output/'vectors.npz')
        atomic_json(output/'records.json', records)
        atomic_json(output/'manifest.json', {'identity': identity,
            'sha256': {name: file_sha256(output/name) for name in ('vectors.npz', 'records.json')}})
    finally:
        temporary.unlink(missing_ok=True)


def reopen(output, identity):
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['identity'] != identity:
        raise ValueError('Centered index identity differs')
    for name in ('vectors.npz', 'records.json'):
        if file_sha256(output/name) != manifest['sha256'][name]:
            raise ValueError('Centered index digest differs')
    with np.load(output/'vectors.npz', allow_pickle=False) as arrays:
        keys, centers = arrays['keys'], arrays['centers']
    records = json.loads((output/'records.json').read_text())
    if (keys.shape != (len(records), identity['width']) or centers.shape != (2, identity['width'])
            or any(x.dtype != np.float32 or not np.isfinite(x).all() for x in (keys, centers))
            or any('terms' in row for row in records)):
        raise ValueError('Invalid centered vectors/metadata')
    return records, keys, centers


def rank(records, keys, query_center, query, *, width, seed, query_time, top_k):
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('Positive integer top_k required')
    counts = DENSE['LEXICAL']['terms'](query)
    if not counts:
        return []
    vector = normalize(DENSE['encode'](counts, width=width, seed=seed)-query_center)
    indices = [i for i, r in enumerate(records)
               if DENSE['eligible'](r, query_time, 'global', 's0', 'frozen-v1', 'research')]
    if not indices or not np.any(vector):
        return []
    scores = keys[indices] @ vector
    return sorted([{'record_id': records[i]['record_id'], 'score': float(score)}
                   for i, score in zip(indices, scores, strict=True)],
                  key=lambda row: (-row['score'], row['record_id']))[:top_k]


def run(calibration, corpus, reference, dense_root, output):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        def check():
            if stop['signal'] is not None or stop_requested(output):
                raise RuntimeError('Centered lexical control stopped; completed arms are reusable')
        check()
        prior = json.loads(reference.read_text())
        base = json.loads((dense_root/'inputs.json').read_text())
        indexed = EpisodeIndex(calibration)
        identity = {'calibration_sha256': indexed.sha256, 'episodes_sha256': file_sha256(corpus),
                    'reference_sha256': file_sha256(reference), 'dense_inputs_sha256': file_sha256(dense_root/'inputs.json'),
                    'script_sha256': file_sha256(__file__), 'dense_script_sha256': file_sha256(DENSE_PATH),
                    'widths': [64, 128, 256, 1024], 'seeds': [11, 23, 47]}
        if (identity['episodes_sha256'] != prior['inputs']['episodes_sha256']
                or identity['reference_sha256'] != base['reference_sha256']
                or identity['dense_script_sha256'] != base['script_sha256']):
            raise ValueError('Frozen dense reference differs')
        if (output/'inputs.json').exists() and json.loads((output/'inputs.json').read_text()) != identity:
            raise ValueError('Centered lexical inputs changed')
        ensure_free(output, 10*1024**3)
        atomic_json(output/'inputs.json', identity)
        calibration_episodes = list(indexed)
        calibration_sources = {s.record_id for e in calibration_episodes for s in e.supports}
        learned = {r['episode']: r for r in prior['rows'] if r['condition'] == 'all'}
        all_episodes = load_episodes(corpus)
        bank_sources = {s.record_id for e in all_episodes for s in e.supports}
        if calibration_sources & bank_sources or {e.episode_id for e in calibration_episodes} & learned.keys():
            raise ValueError('Calibration overlaps evaluation source/query IDs')
        episodes = [e for e in all_episodes if e.episode_id in learned]
        if {e.episode_id for e in episodes} != learned.keys():
            raise ValueError('Reference query set differs')
        for width in identity['widths']:
            for seed in identity['seeds']:
                check()
                label = f'width-{width}-seed-{seed}'
                destination = output/(label+'.json')
                if destination.exists():
                    if json.loads(destination.read_text())['inputs'] != identity:
                        raise ValueError('Completed result identity differs')
                    continue
                records, keys, settings = DENSE['reopen'](dense_root/label)
                source_result = json.loads((dense_root/(label+'.json')).read_text())
                if (settings != {'width': width, 'seed': seed} or source_result['inputs'] != base
                        or file_sha256(dense_root/label/'manifest.json') != source_result['index_manifest_sha256']):
                    raise ValueError('Source dense key artifact differs')
                centers = calibrate(calibration_episodes, width=width, seed=seed)
                path = output/label
                index_identity = identity | settings | {'source_index_manifest_sha256': source_result['index_manifest_sha256']}
                ensure_free(output, 10*1024**3)
                publish(path, records, keys, centers, index_identity)
                records, keys, centers = reopen(path, index_identity)
                rows = []
                for episode in episodes:
                    check()
                    old = learned[episode.episode_id]
                    if (old['environment'], old['task_family'], old['answer']) != (
                            episode.environment, episode.task_family, episode.answer):
                        raise ValueError('Reference query identity differs')
                    ranked = rank(records, keys, centers[1], episode.query, **settings,
                                  query_time=episode.query_time, top_k=2)
                    for count in (1, 2):
                        selected = [r['record_id'] for r in ranked[:count]]
                        rows.append({'episode': episode.episode_id, 'environment': episode.environment,
                            'task_family': episode.task_family, 'top_k': count, 'selected_ids': selected,
                            **DENSE['LEXICAL']['support_metrics'](episode, selected)})
                summary = defaultdict(dict)
                for family in sorted({r['task_family'] for r in rows}):
                    for count in (1, 2):
                        group = [r for r in rows if r['task_family'] == family and r['top_k'] == count]
                        summary[family][str(count)] = {'n': len(group),
                            'complete_support': sum(r['complete_support'] for r in group),
                            'all_required': sum(r['all_required'] for r in group)}
                result = {'inputs': identity, **settings, 'key_blob_bytes': keys.nbytes,
                    'calibration_vector_bytes': centers.nbytes,
                    'serialized_index_bytes': sum(p.stat().st_size for p in path.iterdir()),
                    'index_manifest_sha256': file_sha256(path/'manifest.json'),
                    'calibration_episodes': len(calibration_episodes), 'calibration_sources': len(calibration_sources),
                    'summary': summary, 'rows': rows,
                    'notice': 'Post-hoc centering follow-up using disjoint prior training source/query content. '
                              'All declared widths/seeds reported; literal-name fixture, no generation or capacity claim.'}
                _atomic_text(destination, json.dumps(result, indent=2)+'\n')
                print(json.dumps({'completed': label, 'summary': summary}), flush=True)
        if file_sha256(calibration) != identity['calibration_sha256'] or file_sha256(corpus) != identity['episodes_sha256']:
            raise ValueError('Calibration/evaluation corpus changed during the control')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('calibration', 'corpus', 'reference', 'dense-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.calibration, args.corpus, args.reference, args.dense_root, args.output)
