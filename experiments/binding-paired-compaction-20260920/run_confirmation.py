"""Verify fixed control/source identities, then use the bounded confirmation queue."""
import argparse
import json
from pathlib import Path
import runpy
import signal

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import run_lock
from sdkb.trajectories import file_sha256


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads((args.root/'inputs.json').read_text())
    for path, expected in [(Path(inputs['source'])/'manifest.json', inputs['source_manifest_sha256']),
                           (Path(inputs['train']), inputs['train_sha256']),
                           (resolve_checkpoint(args.root/'interleaved', verify=True)/'manifest.json',
                            inputs['control_manifest_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError(f'Fixed comparison input changed: {path}')
    queue = runpy.run_path(str(Path(__file__).parents[1]/'binding-compact-aware-20260920/run_confirmation.py'))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    with run_lock(args.root/'confirmation-queue', clear_stop=False):
        queue['main'](args.root)
