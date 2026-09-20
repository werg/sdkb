"""Wait for primary confirmation, then run the sealed fit/text diagnostics."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import signal
import time

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import run_lock
from sdkb.trajectories import file_sha256
import sdkb


def run(root, frozen):
    if (Path(sdkb.__file__).resolve().parent != (frozen/'src/sdkb').resolve()
            or Path(os.environ.get('GIT_WORK_TREE', '.')).resolve() != frozen.resolve()):
        raise ValueError('Launch with the frozen model checkout PYTHONPATH and Git environment')
    helper = Path(__file__).parents[1]/'binding-compact-aware-20260920/run_confirmation.py'
    queue = runpy.run_path(str(helper))
    signal.signal(signal.SIGTERM, queue['stop'])
    signal.signal(signal.SIGINT, queue['stop'])
    inputs = json.loads((root/'inputs.json').read_text())
    declared = Path(__file__).with_name('training-diagnostic-inputs.json')
    if file_sha256(root/'training-diagnostic/inputs.json') != file_sha256(declared):
        raise ValueError('Supplemental declaration changed')
    fit = json.loads(declared.read_text())
    if (file_sha256(root/'training-diagnostic/episodes.jsonl') != fit['episodes_sha256']
            or file_sha256(root/'heldout.jsonl') != inputs['heldout_sha256']):
        raise ValueError('Confirmation corpus changed')
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
        if (file_sha256(root/'training-diagnostic/episodes.jsonl') != fit['episodes_sha256']
                or file_sha256(root/'heldout.jsonl') != inputs['heldout_sha256']):
            raise ValueError('Confirmation corpus changed while waiting')
        endpoints = {}
        for size in inputs['config_sha256']:
            checkpoint = resolve_checkpoint(root/f'worlds-{size}', verify=True)
            if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
                raise ValueError('Training is incomplete')
            path = root/'confirmation'/f'worlds-{size}'/'results.json'
            primary = json.loads(path.read_text())
            if (primary['inputs']['checkpoint_manifest_sha256'] != file_sha256(checkpoint/'manifest.json')
                    or primary['inputs']['episodes_sha256'] != inputs['heldout_sha256']
                    or primary['inputs']['script_sha256'] != file_sha256(frozen/'scripts/evaluate_oracle_transfer.py')):
                raise ValueError('Primary confirmation identity changed')
            endpoints[size] = checkpoint
        print(json.dumps({'phase': 'supplemental', 'controller_sha256': file_sha256(__file__),
                          'queue_sha256': file_sha256(helper), 'frozen_model_code': str(frozen)}), flush=True)
        training_jobs = [(f'worlds-{size}-training', ['--source', str(checkpoint),
            '--episodes', str(root/'training-diagnostic/episodes.jsonl'),
            '--output', str(root/'confirmation'/f'worlds-{size}-training')])
            for size, checkpoint in endpoints.items()]
        queue['run_jobs'](root, training_jobs, script=frozen/'scripts/evaluate_oracle_transfer.py')
        text_jobs = [(f'worlds-{size}-text', ['--source', str(checkpoint),
            '--episodes', str(root/'heldout.jsonl'),
            '--reference', str(root/'confirmation'/f'worlds-{size}'/'results.json'),
            '--output', str(root/'confirmation'/f'worlds-{size}-text')])
            for size, checkpoint in endpoints.items()]
        queue['run_jobs'](root, text_jobs,
            script=frozen/'experiments/binding-fixed-query-detail-20260920/text_control.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frozen-model-code', type=Path, required=True)
    args = parser.parse_args()
    # Launch with this frozen checkout's PYTHONPATH and Git environment as well;
    # children inherit it. The controller itself is pinned separately.
    run(args.root, args.frozen_model_code)
