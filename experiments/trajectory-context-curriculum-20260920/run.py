"""Train a source-utility-selected Muon fork with exact external recovery."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path

import yaml

from sdkb.checkpoints import resolve_checkpoint
from sdkb.config import load_config
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.probes import model_probe
from sdkb.training import train
from sdkb.trajectories import file_sha256


def _select_lines(lines: list[str], gains: dict[str, float]) -> tuple[list[str], set[str]]:
    """Select the upper half by a training-only score; copy original lines intact."""
    records = [json.loads(line) for line in lines]
    ids = [record['episode_id'] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate episode identities in original training data')
    if set(ids) != set(gains) or any(not math.isfinite(value) for value in gains.values()):
        raise ValueError('Teacher score identities or values differ from training data')
    selected = set(sorted(ids, key=lambda episode: (-gains[episode], episode))[:len(ids) // 2])
    return [line for line, episode in zip(lines, ids, strict=True) if episode in selected], selected


def prepare(root: Path, *, scores: Path, parent: Path, base_config: Path,
            train_file: Path, validation: Path) -> dict:
    if not root.is_dir() or root.stat().st_dev == Path(__file__).resolve().parents[2].stat().st_dev:
        raise ValueError('Create the run root on an external mounted disk first')
    if (root / 'inputs.json').exists() or (root / 'selected-train.jsonl').exists():
        raise ValueError('Study already prepared; use its locked inputs')
    checkpoint = resolve_checkpoint(parent, verify=True)
    parent_manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if parent_manifest['step'] != 400 or parent_manifest['dataset_sha256'] != file_sha256(train_file):
        raise ValueError('Expected completed joint parent and its original training episodes')
    score = json.loads(scores.read_text())
    if (score['inputs']['episodes_sha256'] != file_sha256(train_file)
            or score['inputs']['checkpoint_model_sha256'] != file_sha256(checkpoint / 'model.safetensors')
            or score['inputs']['depths'] != [1]):
        raise ValueError('Training-only source-utility score inputs differ')
    by_episode = defaultdict(dict)
    for row in score['rows']:
        if row['condition'] in by_episode[row['episode']]:
            raise ValueError('Duplicate source-utility score')
        by_episode[row['episode']][row['condition']] = row
    if any(set(values) != {'r1_selected_text', 'r1_none'} for values in by_episode.values()):
        raise ValueError('Incomplete selected/no-text training score')
    gains = {episode: rows['r1_none']['mean_nll'] - rows['r1_selected_text']['mean_nll']
             for episode, rows in by_episode.items()}
    with train_file.open(encoding='utf-8', newline='') as stream:
        lines = stream.readlines()
    selected_lines, selected = _select_lines(lines, gains)
    if not selected_lines or len(selected) != len(lines) // 2:
        raise ValueError('Invalid upper-half training selection')
    selected_file = root / 'selected-train.jsonl'
    with selected_file.open('x', encoding='utf-8', newline='') as stream:
        stream.writelines(selected_lines)
    config = load_config(base_config)
    if (config.train.oracle_distillation_weight or config.train.oracle_anchor_weight
            or config.model.loops != 2 or config.model.writer_loops != 1
            or not config.model.freeze_backbone or config.train.optimizer != 'muon'
            or config.train.episodes_file != str(train_file)):
        raise ValueError('Base control config differs from the matched native Muon arm')
    config.train.episodes_file = str(selected_file)
    config.train.wandb_group = 'trajectory-context-curriculum-20260920'
    config.validate()
    config_path = root / 'selected-context.yaml'
    config_path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    trajectories = {name: {by_episode[episode]['r1_none']['trajectory'] for episode in ids}
                    for name, ids in (('selected', selected), ('omitted', set(gains) - selected))}
    lock = dict(parent=str(parent), parent_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
        parent_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
        source_train=str(train_file), source_train_sha256=file_sha256(train_file),
        selected_train=str(selected_file), selected_train_sha256=file_sha256(selected_file),
        score_file=str(scores), score_sha256=file_sha256(scores),
        validation=str(validation), validation_sha256=file_sha256(validation),
        config_sha256=file_sha256(config_path), steps=config.train.steps,
        selected_episodes=len(selected), omitted_episodes=len(gains) - len(selected),
        selected_trajectories=len(trajectories['selected']), omitted_trajectories=len(trajectories['omitted']),
        mean_text_gain_selected=sum(gains[i] for i in selected) / len(selected),
        mean_text_gain_omitted=sum(gains[i] for i in gains if i not in selected) / (len(gains) - len(selected)),
        policy='Upper half of original training episodes by fixed-parent R=1 selected-text target NLL gain; stable ID tie-break; original episode lines unchanged')
    atomic_json(root / 'inputs.json', lock)
    return lock


def run(root: Path) -> dict:
    with run_lock(root, clear_stop=True):
        lock = json.loads((root / 'inputs.json').read_text())
        for path, expected in ((Path(lock['source_train']), lock['source_train_sha256']),
                               (Path(lock['selected_train']), lock['selected_train_sha256']),
                               (Path(lock['score_file']), lock['score_sha256']),
                               (Path(lock['validation']), lock['validation_sha256']),
                               (root / 'selected-context.yaml', lock['config_sha256'])):
            if file_sha256(path) != expected:
                raise ValueError(f'Context-curriculum input changed: {path}')
        checkpoint = resolve_checkpoint(lock['parent'], verify=True)
        if (file_sha256(checkpoint / 'manifest.json') != lock['parent_manifest_sha256']
                or file_sha256(checkpoint / 'model.safetensors') != lock['parent_model_sha256']):
            raise ValueError('Context-curriculum parent changed')
        config = load_config(root / 'selected-context.yaml')
        if config.train.episodes_file != lock['selected_train'] or config.train.steps != lock['steps']:
            raise ValueError('Selected data or update budget differs from lock')
        if stop_requested(root):
            return {'status': 'stopped_before_model'}
        probe = root / 'model-probe.json'
        if not probe.exists():
            atomic_json(probe, model_probe(config))
        stage = root / 'selected_context'
        existing = stage.exists()
        if existing:
            manifest = json.loads((resolve_checkpoint(stage, verify=True) / 'manifest.json').read_text())
            if (manifest['dataset_sha256'] != lock['selected_train_sha256']
                    or manifest['step'] > lock['steps']):
                raise ValueError('Existing curriculum state differs from lock')
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
    parser.add_argument('--scores', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--train-file', type=Path)
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    if args.prepare:
        if any(value is None for value in (args.scores, args.parent, args.base_config,
                                           args.train_file, args.validation)):
            parser.error('--prepare requires scores, parent, base config, train file and validation')
        result = prepare(args.root, scores=args.scores, parent=args.parent,
                         base_config=args.base_config, train_file=args.train_file,
                         validation=args.validation)
    else:
        result = run(args.root)
    print(json.dumps(result), flush=True)
