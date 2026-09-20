"""Select frequently seen targets to distinguish training fit from generalization."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def prepare(source, output, report):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '6c6bb47ad7e30f42992b859316124c795832650542d2f28fcfbb9256a071e389':
        raise ValueError('Expected the sealed fixed-query training corpus')
    rows = [json.loads(line) for line in raw.splitlines()]
    rng, counts, families = random.Random(79), Counter(), Counter()
    for _step in range(1600):
        rng.choice([2, 3])
        for _micro in range(4):
            episode = rng.choice(rows)
            families[episode['task_family']] += 1
            if episode['task_family'] == 'multiuse/identifier':
                counts[episode['episode_id']] += 1
            for _source in episode['supports']:
                rng.random()  # Main trainer consumes live/cached choice even at fraction 1.
    if len(counts) != 941 or families['multiuse/identifier'] != 1285:
        raise ValueError('Reconstructed sampling differs from the recorded training endpoint')
    selected = sorted(counts, key=lambda eid: (-counts[eid], eid))[:64]
    by_id = {e['episode_id']: e for e in rows}
    data = ''.join(json.dumps(by_id[eid], sort_keys=True)+'\n' for eid in selected).encode()
    if output.exists():
        if output.read_bytes() != data:
            raise ValueError('Existing training-fit subset differs')
    else:
        output.write_bytes(data)
    value = {
        'training_sha256': hashlib.sha256(raw).hexdigest(),
        'subset_sha256': hashlib.sha256(data).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sampling_seed': 79, 'steps': 1600, 'microbatches_per_step': 4,
        'selection': '64 most frequently sampled identifier episode IDs; descending exposure count then opaque ID. '
                     'Post-hoc training-fit diagnostic, not held-out evaluation.',
        'selected_exposure_histogram': dict(Counter(counts[eid] for eid in selected)),
        'identifier_microbatches': 1285, 'distinct_identifier_queries_sampled': 941,
        'unique_selected_worlds': len({by_id[eid]['environment'] for eid in selected}),
    }
    report.write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.report)
