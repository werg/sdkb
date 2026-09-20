"""Bounded offline-bank and frozen-readout queue with explicit resume controls."""
import argparse
import json
from pathlib import Path
import runpy
import signal

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import run_lock
from sdkb.trajectories import file_sha256


def stages(root, inputs):
    return [root/label/name for label in inputs['models']
            for name in ('bank', *inputs['representations'])]


def run(root, *, resume=False):
    code = Path(__file__).parents[2]
    queue = runpy.run_path(str(code/'experiments/binding-compact-aware-20260920/run_confirmation.py'))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    inputs = json.loads((root/'inputs.json').read_text())
    if file_sha256(root/'episodes.jsonl') != inputs['episodes_sha256']:
        raise ValueError('Readout corpus changed')
    for model in inputs['models'].values():
        checkpoint = resolve_checkpoint(model['checkpoint'], verify=True)
        if file_sha256(checkpoint/'manifest.json') != model['manifest_sha256']:
            raise ValueError('Readout source model changed')
    with run_lock(root/'confirmation-queue', clear_stop=resume):
        if resume:
            # Explicit resume acknowledges control files only after the prior
            # owners release their locks, including signal-stopped probe children.
            for path in (root, *stages(root, inputs)):
                with run_lock(path):
                    pass
        queue['check_stop'](root)
        jobs = [(label+'-bank', ['--source', model['checkpoint'],
                 '--episodes', str(root/'episodes.jsonl'), '--output', str(root/label/'bank')])
                for label, model in inputs['models'].items()]
        queue['run_jobs'](root, jobs, script=code/'scripts/build_frozen_bank.py')
        for representation in inputs['representations']:
            jobs = [(label+'-'+representation, ['--source', model['checkpoint'],
                '--episodes', str(root/'episodes.jsonl'), '--bank', str(root/label/'bank/bank.sqlite'),
                '--output', str(root/label/representation), '--steps', str(inputs['steps']),
                '--heldout-worlds', str(inputs['heldout_worlds']), '--representation', representation])
                for label, model in inputs['models'].items()]
            queue['run_jobs'](root, jobs, script=code/'scripts/probe_payload_identifiers.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, resume=args.resume)
