"""Run the locked recurrent joint stage from the verified latent checkpoint."""
import argparse
import json
from pathlib import Path

from sdkb.checkpoints import resolve_checkpoint, stop_on_signal
from sdkb.config import load_config
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.probes import model_probe
from sdkb.training import train
from sdkb.trajectories import file_sha256


def run(root, *, resume=False):
    inputs = json.loads((root / 'joint-inputs.json').read_text())
    for path, expected in ((root / 'recurrent_joint.yaml', inputs['config_sha256']),
                           (Path(inputs['train']), inputs['train_sha256']),
                           (Path(inputs['validation']), inputs['validation_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError('Declared joint input changed')
    config = load_config(root / 'recurrent_joint.yaml')
    if (config.model.revision != inputs['revision'] or config.train.steps != inputs['steps']
            or config.train.episodes_file != inputs['train'] or config.train.arm != 'memory'
            or config.model.loops != 3 or config.model.writer_loops != 1
            or config.train.loop_counts != [2, 3] or config.model.backbone_train_scope != 'recurrent_core'
            or config.train.oracle_anchor_weight != 0.1 or config.train.oracle_anchor_loops != 1):
        raise ValueError('Joint revision, depth, writer, anchor or budget differs')
    source = resolve_checkpoint(root / 'latent_warmup', verify=True)
    if (file_sha256(source / 'manifest.json') != inputs['latent_manifest_sha256']
            or file_sha256(source / 'model.safetensors') != inputs['latent_model_sha256']
            or json.loads((source / 'manifest.json').read_text())['step'] != 400):
        raise ValueError('Latent initialization changed')
    with run_lock(root, clear_stop=resume), stop_on_signal() as signals:
        if signals['signal'] or stop_requested(root):
            return
        probe = root / 'joint-model-probe.json'
        if not probe.exists():
            atomic_json(probe, model_probe(config))
        if signals['signal'] or stop_requested(root):
            return
        stage = root / 'recurrent_joint'
        existing = stage.exists()
        if existing:
            if not resume:
                raise ValueError('Existing joint stage requires --resume')
            with run_lock(stage, clear_stop=False):
                checkpoint = resolve_checkpoint(stage, verify=True)
                manifest = json.loads((checkpoint / 'manifest.json').read_text())
                if manifest['dataset_sha256'] != inputs['train_sha256']:
                    raise ValueError('Joint checkpoint dataset differs')
                if manifest['step'] > inputs['steps']:
                    raise ValueError('Joint checkpoint exceeds declared budget')
                if manifest['step'] == inputs['steps']:
                    return
        result = train(config, stage, resume=existing,
                       init_from=None if existing else root / 'latent_warmup', stop_output=root)
        atomic_json(root / 'joint-training-result.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, resume=args.resume)
