"""Verify the declared real-trajectory bridge inputs, preflight, then train with recovery."""
import argparse
import json
from pathlib import Path

from sdkb.checkpoints import stop_on_signal
from sdkb.config import load_config
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.probes import model_probe
from sdkb.training import train
from sdkb.trajectories import file_sha256


def run(root, *, resume=False):
    inputs = json.loads((root/'inputs.json').read_text())
    for path, expected in [(root/'recurrence_bridge.yaml', inputs['config_sha256']),
                           (inputs['train'], inputs['train_sha256']),
                           (inputs['validation'], inputs['validation_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError('Declared bridge input changed')
    config = load_config(root/'recurrence_bridge.yaml')
    if config.model.revision != inputs['revision'] or config.train.steps != inputs['steps']:
        raise ValueError('Bridge revision/budget differs')
    with run_lock(root, clear_stop=resume), stop_on_signal() as signals:
        if signals['signal'] or stop_requested(root):
            return
        probe = root/'model-probe.json'
        if not probe.exists():
            atomic_json(probe, model_probe(config))
        if signals['signal'] or stop_requested(root):
            return
        result = train(config, root/'recurrence_bridge', resume=resume, stop_output=root)
        atomic_json(root/'training-result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, resume=args.resume)
