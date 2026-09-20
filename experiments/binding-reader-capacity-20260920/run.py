"""Controlled reader-width forks followed by stored-only confirmation."""
import argparse
import json
from pathlib import Path
import runpy
import signal

from sdkb.checkpoints import resolve_checkpoint
from sdkb.config import load_config
from sdkb.operations import run_lock
from sdkb.training import train
from sdkb.trajectories import file_sha256


def verify(root):
    inputs = json.loads((root/'inputs.json').read_text())
    paths = [(inputs['source']+'/manifest.json', inputs['source_manifest_sha256']),
             (inputs['episodes'], inputs['episodes_sha256']),
             (inputs['heldout'], inputs['heldout_sha256'])]
    paths.extend((root/(name+'.yaml'), value) for name, value in inputs['configs'].items())
    for path, expected in paths:
        if file_sha256(path) != expected:
            raise ValueError(f'Capacity study input changed: {path}')
    return inputs


def run(root, resume=False, arm=None):
    inputs = verify(root)
    if arm is not None:
        if arm not in inputs['configs']:
            raise ValueError('Unknown training arm')
        config = load_config(root/(arm+'.yaml'))
        output = root/arm
        existing = output.exists()
        if existing and not resume:
            raise ValueError('Existing training output requires --resume')
        result = train(config, output, resume=existing,
                       init_from=None if existing else inputs['source'], stop_output=root)
        if result['stopped_early'] or result['stop_requested']:
            raise RuntimeError('Training stopped with recovery state preserved')
        return
    code = Path(__file__).parents[2]
    queue = runpy.run_path(str(code/'experiments/binding-compact-aware-20260920/run_confirmation.py'))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    with run_lock(root/'confirmation-queue', clear_stop=resume):
        if resume:
            for path in (root, *(root/name for name in inputs['configs']),
                         *(root/'confirmation'/name for name in inputs['configs'])):
                with run_lock(path):
                    pass
        queue['check_stop'](root)
        jobs = []
        for name in inputs['configs']:
            stage = root/name
            if resume and stage.exists():
                with run_lock(stage, clear_stop=False):
                    cp = resolve_checkpoint(stage, verify=True)
                    if json.loads((cp/'manifest.json').read_text())['step'] == inputs['steps']:
                        continue
            jobs.append((name, ['--root', str(root), '--arm', name] + (['--resume'] if resume else [])))
        queue['run_jobs'](root, jobs, script=Path(__file__))
        evaluations = []
        for name in inputs['configs']:
            checkpoint = resolve_checkpoint(root/name, verify=True)
            if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
                raise ValueError('Incomplete training endpoint')
            evaluations.append((name+'-heldout', ['--source', str(checkpoint), '--episodes', inputs['heldout'],
                '--output', str(root/'confirmation'/name)]))
        queue['run_jobs'](root, evaluations, script=code/'scripts/evaluate_oracle_transfer.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--arm')
    args = parser.parse_args()
    run(args.root, args.resume, args.arm)
