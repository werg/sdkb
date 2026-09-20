"""Run a matched native Muon fork with an explicitly opened memory gate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from sdkb.checkpoints import resolve_checkpoint
from sdkb.config import load_config
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.probes import model_probe
from sdkb.training import train
from sdkb.trajectories import file_sha256


MEMORY_GATE = .30


def prepare(root: Path, *, parent: Path, base_config: Path,
            train_file: Path, validation: Path) -> dict:
    if not root.is_dir() or root.stat().st_dev == Path(__file__).resolve().parents[2].stat().st_dev:
        raise ValueError('Create the study root on an external mounted disk')
    if any(root.iterdir()):
        raise ValueError('Fresh gate study requires an empty external root')
    checkpoint = resolve_checkpoint(parent, verify=True)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if manifest['step'] != 400 or manifest['dataset_sha256'] != file_sha256(train_file):
        raise ValueError('Expected the completed joint parent and its training data')
    config = load_config(base_config)
    if (config.train.episodes_file != str(train_file) or config.train.seed != 233
            or config.train.steps != 400 or config.train.optimizer != 'muon'
            or config.train.oracle_distillation_weight or config.train.oracle_anchor_weight
            or config.model.loops != 2 or config.model.writer_loops != 1
            or not config.model.freeze_backbone or config.train.warmstart_memory_gate is not None):
        raise ValueError('Base config differs from the matched uniform-data control')
    config.train.warmstart_memory_gate = MEMORY_GATE
    config.train.wandb_group = 'trajectory-memory-gate-20260920'
    config.validate()
    config_path = root / 'memory-gate.yaml'
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    lock = dict(parent=str(parent), parent_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
                parent_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
                base_config=str(base_config), base_config_sha256=file_sha256(base_config),
                config_sha256=file_sha256(config_path), train_file=str(train_file),
                train_sha256=file_sha256(train_file), validation=str(validation),
                validation_sha256=file_sha256(validation), steps=config.train.steps,
                memory_gate=MEMORY_GATE, comparison='Same parent, train data, seed, depth and Muon as no-KL control; initial bridge memory gate only')
    atomic_json(root / 'inputs.json', lock)
    return lock


def run(root: Path) -> dict:
    with run_lock(root, clear_stop=True):
        lock = json.loads((root / 'inputs.json').read_text())
        for path, expected in ((Path(lock['base_config']), lock['base_config_sha256']),
                               (root / 'memory-gate.yaml', lock['config_sha256']),
                               (Path(lock['train_file']), lock['train_sha256']),
                               (Path(lock['validation']), lock['validation_sha256'])):
            if file_sha256(path) != expected:
                raise ValueError(f'Gate study input changed: {path}')
        checkpoint = resolve_checkpoint(lock['parent'], verify=True)
        if (file_sha256(checkpoint / 'manifest.json') != lock['parent_manifest_sha256']
                or file_sha256(checkpoint / 'model.safetensors') != lock['parent_model_sha256']):
            raise ValueError('Gate study parent changed')
        config = load_config(root / 'memory-gate.yaml')
        if (config.train.warmstart_memory_gate != lock['memory_gate']
                or config.train.steps != lock['steps'] or config.train.episodes_file != lock['train_file']):
            raise ValueError('Gate, budget or training episodes differ from lock')
        if stop_requested(root):
            return {'status': 'stopped_before_model'}
        probe = root / 'model-probe.json'
        if not probe.exists():
            atomic_json(probe, model_probe(config))
        stage = root / 'memory_gate'
        existing = stage.exists()
        if existing:
            manifest = json.loads((resolve_checkpoint(stage, verify=True) / 'manifest.json').read_text())
            if manifest['dataset_sha256'] != lock['train_sha256'] or manifest['step'] > lock['steps']:
                raise ValueError('Existing gate checkpoint differs from locked study')
            if manifest['step'] == lock['steps']:
                return {'status': 'complete', 'steps': lock['steps']}
        result = train(config, stage, resume=existing,
                       init_from=None if existing else lock['parent'], stop_output=root)
        atomic_json(root / 'training-result.json', result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--train-file', type=Path)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if any(value is None for value in (args.parent, args.base_config, args.train_file, args.validation)):
            parser.error('--prepare needs parent, base config, train file and validation')
        result = prepare(args.root, parent=args.parent, base_config=args.base_config,
                         train_file=args.train_file, validation=args.validation)
    else:
        result = run(args.root)
    print(json.dumps(result), flush=True)
