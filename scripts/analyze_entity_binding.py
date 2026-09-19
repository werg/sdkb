"""Stratify a completed two-entity world-context evaluation; no model execution."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def analyze(episodes_path: Path, results_path: Path):
    episodes_bytes, results_bytes = episodes_path.read_bytes(), results_path.read_bytes()
    episodes = [json.loads(line) for line in episodes_bytes.splitlines() if line.strip()]
    by_id = {e['episode_id']: e for e in episodes}
    worlds = defaultdict(dict)
    for episode in episodes:
        if episode['task_family'] == 'multiuse/action':
            worlds[episode['environment']][tuple(sorted(episode['required_ids']))] = episode
    if not worlds or any(len(entities) != 2 for entities in worlds.values()):
        raise ValueError('This diagnostic requires exactly two entities per world')
    groups, pairs = defaultdict(list), defaultdict(list)
    for row in json.loads(results_bytes)['rows']:
        family = row['task_family']
        if row['condition'] != 'all' or family not in {
                'multiuse/action', 'multiuse/permission', 'multiuse/restoration'}:
            continue
        episode = by_id[row['episode']]
        entities = worlds[episode['environment']]
        mine = [e for ids, e in entities.items() if set(episode['required_ids']) & set(ids)]
        other = [e for ids, e in entities.items() if not set(episode['required_ids']) & set(ids)]
        if len(mine) != 1 or len(other) != 1:
            raise ValueError('Cannot identify the target and competing entity')
        mine, other = mine[0], other[0]
        if family == 'multiuse/action':
            group = (f"permission_same={mine['allowed_capability'] == other['allowed_capability']},"
                     f"restoration_same={mine['restore'] == other['restore']}")
        else:
            field = 'allowed_capability' if family.endswith('permission') else 'restore'
            group = 'same_rule' if mine[field] == other[field] else 'opposed_rule'
            if group == 'opposed_rule':
                pairs[(family, episode['environment'])].append(row)
        groups[(family, group)].append(row['choice_correct'])
    paired = {}
    for family in ('multiuse/permission', 'multiuse/restoration'):
        selected = [rows for (f, _), rows in pairs.items() if f == family]
        if any(len(rows) != 2 for rows in selected):
            raise ValueError('Expected one fact question for each of two entities')
        paired[family] = {
            'opposed_worlds': len(selected),
            'same_prediction_for_both_entities': sum(
                p[0]['predicted_action'] == p[1]['predicted_action'] for p in selected),
            'both_correct': sum(all(row['choice_correct'] for row in p) for p in selected),
        }
    return {
        'diagnostic': 'Post-hoc descriptive stratification of a completed two-entity evaluation.',
        'episodes_sha256': hashlib.sha256(episodes_bytes).hexdigest(),
        'results_sha256': hashlib.sha256(results_bytes).hexdigest(),
        'by_family': {f: {g: {'n': len(v), 'correct': sum(v), 'accuracy': sum(v) / len(v)}
                         for (family, g), v in groups.items() if family == f}
                      for f in sorted({k[0] for k in groups})},
        'entity_pairs': paired,
        'notice': 'Agreement cases can be solved without entity discrimination. '
                  'Inspect opposed-rule pairs before claiming binding; these are not independent training seeds.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.episodes, args.results)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
