"""Compare full results.json files on the same held-out questions."""
import argparse
import hashlib
import json
from pathlib import Path

from sdkb.comparison import compare_transfer_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: path.read_bytes() for name, path in
              [('baseline', args.baseline), ('candidate', args.candidate)]}
    report = compare_transfer_results(*(json.loads(inputs[n]) for n in ('baseline', 'candidate')))
    report['inputs'] = {name: {'path': str(getattr(args, name)),
                              'sha256': hashlib.sha256(data).hexdigest()}
                        for name, data in inputs.items()}
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
