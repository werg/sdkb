"""Wait for primary confirmations, then run the declared frozen text controls."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import signal
import time

import sdkb
from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import run_lock
from sdkb.trajectories import file_sha256


def run(root, frozen):
    if (Path(sdkb.__file__).resolve().parent != (frozen/'src/sdkb').resolve()
            or Path(os.environ.get('GIT_WORK_TREE', '.')).resolve() != frozen.resolve()):
        raise ValueError('Use the frozen model PYTHONPATH and Git environment')
    inputs = json.loads((root/'inputs.json').read_text())
    declared = root/'text-controls.json'
    declaration_hash = file_sha256(declared)
    text = json.loads(declared.read_text())
    helper = Path(__file__).parents[1]/'binding-compact-aware-20260920/run_confirmation.py'
    queue = runpy.run_path(str(helper))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    script = frozen/text['script']
    if (file_sha256(script) != text['script_sha256'] or set(text['arms']) != set(inputs['configs'])
            or text['episodes_sha256'] != inputs['heldout_sha256']):
        raise ValueError('Text protocol differs')
    with ExitStack() as ownership:
        while True:
            queue['check_stop'](root)
            try:
                ownership.enter_context(run_lock(root/'confirmation-queue', clear_stop=False))
                break
            except RuntimeError as error:
                if 'already running' not in str(error):
                    raise
            time.sleep(30)
        if file_sha256(declared) != declaration_hash or file_sha256(text['episodes']) != text['episodes_sha256']:
            raise ValueError('Text inputs changed while waiting')
        jobs = []
        for name in text['arms']:
            checkpoint = resolve_checkpoint(root/name, verify=True)
            if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
                raise ValueError('Training is incomplete')
            reference = root/'confirmation'/name/'results.json'
            primary = json.loads(reference.read_text())
            if (primary['inputs']['checkpoint_manifest_sha256'] != file_sha256(checkpoint/'manifest.json')
                    or primary['inputs']['episodes_sha256'] != text['episodes_sha256']
                    or primary['inputs']['script_sha256'] != file_sha256(frozen/'scripts/evaluate_oracle_transfer.py')):
                raise ValueError('Primary confirmation identity differs')
            jobs.append((name+'-text', ['--source', str(checkpoint), '--episodes', text['episodes'],
                '--reference', str(reference), '--output', str(root/'confirmation'/(name+'-text'))]))
        print(json.dumps({'controller_sha256': file_sha256(__file__), 'declaration_sha256': declaration_hash,
                          'frozen_model_code': str(frozen), 'phase': 'text_controls'}), flush=True)
        queue['run_jobs'](root, jobs, script=script)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frozen-model-code', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.frozen_model_code)
