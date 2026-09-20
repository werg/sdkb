"""Confirm matched endpoints in both fixed and original identifier-query forms."""
import argparse
import json
from pathlib import Path
import runpy
import signal
import time

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import run_lock, stop_requested
from sdkb.trajectories import file_sha256


def main(root, queue):
    inputs = json.loads((root/'inputs.json').read_text())
    queue['check_stop'](root)
    control = resolve_checkpoint(Path(inputs['control']), verify=True)
    checks = [(control/'manifest.json', inputs['control_manifest_sha256']),
              (root/'train.jsonl', inputs['data']['output_sha256']),
              (root/'fixed.yaml', inputs['config_sha256'])]
    checks.extend((root/('heldout-'+style+'.jsonl'), digest) for style, digest in inputs['heldout'].items())
    for path, expected in checks:
        if file_sha256(path) != expected:
            raise ValueError(f'Changed fixed-query comparison input: {path}')
    stage = root/'fixed'
    while True:
        queue['check_stop'](root)
        if stop_requested(stage):
            raise RuntimeError('Fixed-query training stop requested')
        try:
            with run_lock(stage, clear_stop=False):
                checkpoint = resolve_checkpoint(stage, verify=True)
                if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
                    raise ValueError('Inactive incomplete fixed-query training')
            break
        except RuntimeError as error:
            if 'already running' not in str(error):
                raise
        time.sleep(30)
    jobs = []
    for style in ('fixed', 'original'):
        for name, source in [('control', control), ('fixed', checkpoint)]:
            label = name+'-'+style
            jobs.append((label, ['--source', str(source), '--episodes', str(root/('heldout-'+style+'.jsonl')),
                                '--output', str(root/'confirmation'/label)]))
    queue['run_jobs'](root, jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    queue = runpy.run_path(str(Path(__file__).parents[1]/'binding-compact-aware-20260920/run_confirmation.py'))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    with run_lock(args.root/'confirmation-queue', clear_stop=False):
        main(args.root, queue)
