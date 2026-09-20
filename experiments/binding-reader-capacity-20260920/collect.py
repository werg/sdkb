"""Collect all frozen capacity-study confirmations after verifying coverage."""
import argparse
import json
from pathlib import Path
import runpy

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def collect(root, output):
    code = Path(__file__).parents[2]
    helper = code/'experiments/binding-endpoint-freshness-20260920/summarize.py'
    confirmation = runpy.run_path(str(helper))['confirmation']
    inputs = json.loads((root/'inputs.json').read_text())
    episodes = Path(inputs['heldout'])
    if file_sha256(episodes) != inputs['heldout_sha256']:
        raise ValueError('Held-out corpus changed')
    reports = {}
    for name in inputs['configs']:
        checkpoint = resolve_checkpoint(root/name, verify=True)
        if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
            raise ValueError('Incomplete training endpoint')
        report, _rows = confirmation(root/'confirmation'/name/'results.json', checkpoint, episodes)
        reports[name] = report
    atomic_json(output, {'arms': reports, 'inputs_sha256': file_sha256(root/'inputs.json'),
        'collector_sha256': file_sha256(__file__), 'coverage_helper_sha256': file_sha256(helper),
        'notice': inputs['notice']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
