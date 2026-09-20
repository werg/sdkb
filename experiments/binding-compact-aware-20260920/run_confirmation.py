"""Wait for committed study endpoints, then run at most two frozen evaluators."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import run_lock, stop_requested
from sdkb.trajectories import file_sha256


def stop(signum, _frame):
    raise SystemExit(128 + signum)


def check_stop(root):
    if stop_requested(root) or stop_requested(root/'confirmation-queue'):
        raise RuntimeError('Confirmation stop requested')


def main(root):
    check_stop(root)
    inputs = json.loads((root/'inputs.json').read_text())
    if file_sha256(root/'heldout.jsonl') != inputs['heldout_sha256']:
        raise ValueError('Held-out corpus changed')
    for name, arm in inputs['arms'].items():
        if file_sha256(root/(name+'.yaml')) != arm['config_sha256']:
            raise ValueError(f'Arm config changed: {name}')
    while True:
        check_stop(root)
        ready = True
        for name in inputs['arms']:
            stage = root/name
            if stop_requested(stage):
                raise RuntimeError(f'Training stop requested: {stage}')
            try:
                with run_lock(stage, clear_stop=False):
                    checkpoint = resolve_checkpoint(stage, verify=False)
                    manifest = json.loads((checkpoint/'manifest.json').read_text())
                    if manifest['step'] != inputs['steps']:
                        raise ValueError(f'Inactive incomplete training: {stage}')
            except RuntimeError as error:
                if 'already running' not in str(error):
                    raise
                ready = False
        if ready:
            break
        time.sleep(30)
    jobs = [(name+'-'+method, ['--source', str(root/name), '--episodes', str(root/'heldout.jsonl'),
             '--output', str(root/'confirmation'/(name+'-'+method)), '--compact-method', method])
            for method in ('raw', 'mean', 'trained') for name in inputs['arms']]
    run_jobs(root, jobs)


def run_jobs(root, jobs, *, script='scripts/evaluate_oracle_transfer.py'):
    """Caller owns the queue lock; all confirmations share stop/cleanup behavior."""
    pending = list(jobs)
    children = {}
    try:
        while pending or children:
            check_stop(root)
            while pending and len(children) < 2:
                check_stop(root)
                label, arguments = pending.pop(0)
                with (root/('confirmation-'+label+'-console.log')).open('a') as log:
                    child = subprocess.Popen([sys.executable, str(script),
                        *arguments],
                        stdout=log, stderr=subprocess.STDOUT)
                children[label] = child
                print(json.dumps({'started': label, 'pid': child.pid}), flush=True)
            for label, child in list(children.items()):
                code = child.poll()
                if code is not None:
                    del children[label]
                    print(json.dumps({'finished': label, 'exit_code': code}), flush=True)
                    if code:
                        raise RuntimeError(f'Confirmation failed: {label}, code {code}')
            if children:
                time.sleep(5)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.terminate()
        for child in children.values():
            child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with run_lock(args.root/'confirmation-queue', clear_stop=False):
        main(args.root)
