"""Run the prepared continuation arms sequentially with cooperative stop/resume."""
import argparse
import json
import os
from pathlib import Path
import time

from sdkb.checkpoints import resolve_checkpoint, stop_on_signal
from sdkb.config import load_config
from sdkb.operations import atomic_json, control_dir, run_lock, stop_requested
from sdkb.training import train
from sdkb.trajectories import file_sha256


def run(root: Path, resume: bool):
    inputs = json.loads((root / 'inputs.json').read_text())
    source = Path(inputs['source_checkpoint'])
    order = inputs.get('order', list(inputs['configs']))
    if len(order) != len(set(order)) or set(order) != set(inputs['configs']):
        raise ValueError('Arm order must include every prepared arm exactly once')
    if file_sha256(source / 'manifest.json') != inputs['source_manifest_sha256']:
        raise ValueError('Source checkpoint identity changed')
    configs = {}
    for arm, record in inputs['configs'].items():
        if file_sha256(record['config']) != record['sha256']:
            raise ValueError(f'Prepared config changed: {arm}')
        config = load_config(record['config'])
        if file_sha256(config.train.episodes_file) != inputs['episodes_sha256']:
            raise ValueError('Training episodes changed')
        if Path(record['run']).exists() and not resume:
            raise FileExistsError('Existing arm requires explicit --resume')
        configs[arm] = config
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(),
             'output': str(root), 'completed_arms': []}
    state_path = control_dir(root) / 'process.json'
    with run_lock(root), stop_on_signal() as stop:
        atomic_json(state_path, state)
        try:
            for arm in order:
                if stop['signal'] is not None or stop_requested(root):
                    state['status'] = 'stopped'
                    break
                output = Path(inputs['configs'][arm]['run'])
                config = configs[arm]
                state['active_arm'] = arm
                atomic_json(state_path, state)
                existing = output.exists()
                if existing:
                    checkpoint = resolve_checkpoint(output, verify=True)
                    saved = json.loads((checkpoint / 'manifest.json').read_text())
                    if saved['step'] == config.train.steps:
                        state['completed_arms'].append(arm)
                        continue
                result = train(config, output, resume=existing,
                               init_from=None if existing else source, stop_output=root)
                if result['stopped_early'] or result['stop_requested']:
                    state['status'] = 'stopped'
                    break
                state['completed_arms'].append(arm)
            else:
                state['status'] = 'complete'
        except BaseException:
            state['status'] = 'failed'
            raise
        finally:
            state['finished_at'] = time.time()
            atomic_json(state_path, state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.study.resolve(), args.resume)
