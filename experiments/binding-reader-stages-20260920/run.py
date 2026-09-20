"""Two bounded intermediate readouts from an existing frozen bank."""
import argparse
import json
from pathlib import Path
import runpy
import signal

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import run_lock
from sdkb.trajectories import file_sha256


def run(root, *, resume=False):
    code = Path(__file__).parents[2]
    queue = runpy.run_path(str(code/'experiments/binding-compact-aware-20260920/run_confirmation.py'))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    inputs = json.loads((root/'inputs.json').read_text())
    script = code/'scripts/probe_payload_identifiers.py'
    for path, expected in [(inputs['episodes'], inputs['episodes_sha256']),
                           (inputs['bank'], inputs['bank_sha256']),
                           (script, inputs['probe_script_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError(f'Readout input changed: {path}')
    checkpoint = resolve_checkpoint(inputs['checkpoint'], verify=True)
    if file_sha256(checkpoint/'manifest.json') != inputs['manifest_sha256']:
        raise ValueError('Frozen source model changed')
    with run_lock(root/'confirmation-queue', clear_stop=resume):
        if resume:
            for path in (root, *(root/name for name in inputs['representations'])):
                with run_lock(path):
                    pass
        queue['check_stop'](root)
        jobs = [(name, ['--source', str(checkpoint), '--episodes', inputs['episodes'],
            '--bank', inputs['bank'], '--output', str(root/name), '--steps', str(inputs['steps']),
            '--heldout-worlds', str(inputs['heldout_worlds']), '--representation', name])
            for name in inputs['representations']]
        queue['run_jobs'](root, jobs, script=script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, resume=args.resume)
