"""Bounded, stoppable source/endpoint confirmation for the copy-fit diagnostic."""
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
    source = resolve_checkpoint(Path(inputs['source']), verify=True)
    for path, expected in [(source/'manifest.json', inputs['source_manifest_sha256']),
                           (root/'train.jsonl', inputs['train_sha256']),
                           (root/'heldout.jsonl', inputs['heldout_sha256']),
                           (root/'copy.yaml', inputs['config_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError(f'Changed copy-fit input: {path}')
    def job(label, checkpoint, episodes):
        return label, ['--source', str(checkpoint), '--episodes', str(root/episodes),
                       '--output', str(root/'confirmation'/label)]
    queue['run_jobs'](root, [job('source-heldout', source, 'heldout.jsonl')])
    stage = root/'copy'
    while True:
        queue['check_stop'](root)
        if stop_requested(stage):
            raise RuntimeError('Copy-fit training stop requested')
        try:
            with run_lock(stage, clear_stop=False):
                checkpoint = resolve_checkpoint(stage, verify=True)
                if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
                    raise ValueError('Inactive incomplete copy-fit training')
            break
        except RuntimeError as error:
            if 'already running' not in str(error):
                raise
        time.sleep(30)
    queue['run_jobs'](root, [job('copy-training', checkpoint, 'train.jsonl'),
                            job('copy-heldout', checkpoint, 'heldout.jsonl')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    queue = runpy.run_path(str(Path(__file__).parents[1]/'binding-compact-aware-20260920/run_confirmation.py'))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    with run_lock(args.root/'confirmation-queue', clear_stop=False):
        main(args.root, queue)
