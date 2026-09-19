"""Wait for the committed endpoint, then evaluate three arms in separate processes."""
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


def main(root):
    inputs = json.loads((root/'inputs.json').read_text())
    for name in ('source', 'control'):
        checkpoint = resolve_checkpoint(Path(inputs[name]), verify=True)
        if file_sha256(checkpoint/'manifest.json') != inputs[name+'_manifest_sha256']:
            raise ValueError(f'Changed {name} checkpoint')
    if file_sha256(root/'heldout.jsonl') != inputs['heldout_sha256']:
        raise ValueError('Changed held-out corpus')
    while True:
        ready = True
        for name in ('views',):
            stage = root/name
            if stop_requested(stage):
                raise RuntimeError(f'Training stop requested: {stage}')
            try:
                with run_lock(stage, clear_stop=False):
                    checkpoint = resolve_checkpoint(stage, verify=False)
                    manifest = json.loads((checkpoint/'manifest.json').read_text())
                    if manifest['step'] != 1600:
                        raise ValueError(f'Inactive incomplete training: {stage} at {manifest["step"]}')
            except RuntimeError as error:
                if 'already running' not in str(error):
                    raise
                ready = False
        if ready:
            break
        time.sleep(30)
    children = []
    try:
        for name, source in [('source', Path(inputs['source'])), ('independent-worlds', Path(inputs['control'])),
                             ('views', root/'views')]:
            with (root/f'confirmation-{name}-console.log').open('a') as log:
                child = subprocess.Popen([sys.executable, 'scripts/evaluate_oracle_transfer.py',
                    '--source', str(source), '--episodes', str(root/'heldout.jsonl'),
                    '--output', str(root/'confirmation'/name)], stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child))
            print(json.dumps({'started_confirmation': name, 'pid': child.pid}), flush=True)
        failures = []
        for name, child in children:
            code = child.wait()
            print(json.dumps({'finished_confirmation': name, 'exit_code': code}), flush=True)
            if code:
                failures.append((name, code))
        if failures:
            raise RuntimeError(f'Confirmation failures: {failures}')
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main(args.root)
