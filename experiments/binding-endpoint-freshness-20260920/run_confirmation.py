"""Bounded, stoppable source/endpoint confirmation for the endpoint freshness study."""
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
    expected_files = [(source/'manifest.json', inputs['source_manifest_sha256']),
                      (root/'heldout.jsonl', inputs['heldout_sha256'])]
    for size, digest in inputs['config_sha256'].items():
        expected_files.append((root/f'worlds-{size}.yaml', digest))
        expected_files.append((root/'data'/f'worlds-{size}.jsonl',
                               inputs['data']['corpora'][size]['sha256']))
    for path, expected in expected_files:
        if file_sha256(path) != expected:
            raise ValueError(f'Changed endpoint freshness input: {path}')
    def job(label, checkpoint, episodes):
        return label, ['--source', str(checkpoint), '--episodes', str(root/episodes),
                       '--output', str(root/'confirmation'/label)]
    queue['run_jobs'](root, [job('source-heldout', source, 'heldout.jsonl')])
    endpoints = {}
    while len(endpoints) < len(inputs['config_sha256']):
        queue['check_stop'](root)
        for size in inputs['config_sha256']:
            if size in endpoints:
                continue
            stage = root/f'worlds-{size}'
            if stop_requested(stage):
                raise RuntimeError(f'Training stop requested: {stage}')
            try:
                with run_lock(stage, clear_stop=False):
                    checkpoint = resolve_checkpoint(stage, verify=True)
                    if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
                        raise ValueError(f'Inactive incomplete training: {stage}')
                    endpoints[size] = checkpoint
            except RuntimeError as error:
                if 'already running' not in str(error):
                    raise
        if len(endpoints) < len(inputs['config_sha256']):
            time.sleep(30)
    queue['run_jobs'](root, [job(f'worlds-{size}', checkpoint, 'heldout.jsonl')
                            for size, checkpoint in endpoints.items()])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    queue = runpy.run_path(str(Path(__file__).parents[1]/'binding-compact-aware-20260920/run_confirmation.py'))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    with run_lock(args.root/'confirmation-queue', clear_stop=False):
        main(args.root, queue)
