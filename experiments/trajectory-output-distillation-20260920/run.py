"""Prepare and run matched fixed-parent Muon forks from the joint trajectory stage.

All mutable outputs, including configs, preflights, W&B and checkpoints, live on
the declared external root. A stopped arm resumes only its own exact checkpoint.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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


PARENT = Path('/archive/runs/trajectory-prefix-muon-20260920/recurrent_joint')
BASE = Path(__file__).parents[1] / 'trajectory-prefix-muon-20260920' / 'recurrent_joint.yaml'
ARMS = {'control': 0.0, 'distilled': 1.0}


def _read_lock(root: Path) -> dict:
    lock = json.loads((root / 'inputs.json').read_text())
    checkpoint = resolve_checkpoint(PARENT, verify=True)
    for path, expected in ((checkpoint / 'manifest.json', lock['parent_manifest_sha256']),
                           (checkpoint / 'model.safetensors', lock['parent_model_sha256']),
                           (Path(lock['train']), lock['train_sha256']),
                           (Path(lock['validation']), lock['validation_sha256'])):
        if file_sha256(path) != expected:
            raise ValueError(f'Distillation input changed: {path}')
    if json.loads((checkpoint / 'manifest.json').read_text())['step'] != 400:
        raise ValueError('Parent is not the completed joint stage')
    for name, expected in lock['config_sha256'].items():
        if file_sha256(root / f'{name}.yaml') != expected:
            raise ValueError(f'Distillation config changed: {name}')
    return lock


def prepare(root: Path, validation: Path) -> dict:
    if not root.is_dir() or root.stat().st_dev == Path('/workspace/sdkb').stat().st_dev:
        raise ValueError('Create the run root on the external mounted disk first')
    if any(root.iterdir()):
        raise ValueError('Preparation requires an empty output root')
    base = load_config(BASE)
    checkpoint = resolve_checkpoint(PARENT, verify=True)
    source_manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if source_manifest['step'] != 400:
        raise ValueError('Parent must be the completed joint stage')
    if file_sha256(Path(base.train.episodes_file)) != source_manifest['dataset_sha256']:
        raise ValueError('Parent and new study train data differ')
    config_hashes = {}
    for name, weight in ARMS.items():
        config = deepcopy(base)
        config.model.loops = 2
        config.model.freeze_backbone = True
        config.train.seed = 233
        config.train.loop_counts = []
        config.train.oracle_anchor_weight = 0.0
        config.train.oracle_alignment_weight = 0.0
        config.train.oracle_distillation_weight = weight
        config.train.wandb_group = 'trajectory-output-distillation-20260920'
        config.validate()
        path = root / f'{name}.yaml'
        path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        config_hashes[name] = file_sha256(path)
    lock = {'parent_checkpoint': str(checkpoint),
            'parent_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
            'parent_model_sha256': file_sha256(checkpoint / 'model.safetensors'),
            'train': base.train.episodes_file,
            'train_sha256': file_sha256(Path(base.train.episodes_file)),
            'validation': str(validation), 'validation_sha256': file_sha256(validation),
            'steps': base.train.steps, 'config_sha256': config_hashes,
            'arms': ARMS, 'comparison': 'same parent, data, seed, R=2 and Muon; output KL weight only'}
    atomic_json(root / 'inputs.json', lock)
    return lock


def run(root: Path) -> None:
    with run_lock(root, clear_stop=True):
        lock = _read_lock(root)
        for name in ARMS:
            if stop_requested(root):
                return
            config = load_config(root / f'{name}.yaml')
            if config.train.steps != lock['steps'] or config.train.oracle_distillation_weight != ARMS[name]:
                raise ValueError('Arm or budget differs from declared study')
            probe = root / f'{name}-model-probe.json'
            if not probe.exists():
                atomic_json(probe, model_probe(config))
            stage = root / name
            existing = stage.exists()
            if existing:
                manifest = json.loads((resolve_checkpoint(stage, verify=True) / 'manifest.json').read_text())
                if manifest['dataset_sha256'] != lock['train_sha256'] or manifest['step'] > lock['steps']:
                    raise ValueError('Existing arm checkpoint differs from study')
                if manifest['step'] == lock['steps']:
                    continue
            result = train(config, stage, resume=existing,
                           init_from=None if existing else PARENT, stop_output=root)
            atomic_json(root / f'{name}-training-result.json', result)
            if result['steps'] != lock['steps']:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.validation is None:
            parser.error('--prepare requires --validation')
        print(json.dumps(prepare(args.root, args.validation)), flush=True)
    else:
        run(args.root)
