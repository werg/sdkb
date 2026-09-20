"""Evaluate learned stored-only retrieval against a published corpus generation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes
from sdkb.evaluation import stored_transfer_evaluation
from sdkb.offline_bank import (assert_bank_writer_compatible, canonical_json,
                               publish_offline_generation)
from sdkb.operations import atomic_json
from sdkb.store import DiskStore
from sdkb.training import config_from_run
from sdkb.trajectories import file_sha256


def evaluate(run: Path, bank_dir: Path, episodes_file: Path, output: Path, *,
             max_episodes: int = 16, limits: tuple[int, ...] = (8, 4, 2, 1)) -> dict:
    if max_episodes < 1 or output.exists() or not output.parent.is_dir():
        raise ValueError('Positive evaluation count and a fresh output parent required')
    config = config_from_run(run)
    training_bank_dir = config.train.bank_dir
    if len(limits) != len(config.memory.payload_dims):
        raise ValueError('One retrieval limit per active space required')
    if any(k < 1 or k > cap for k, cap in zip(limits, config.memory.neighbors, strict=True)):
        raise ValueError('Evaluation limits must fit the model neighborhood caps')
    bank_manifest_path = bank_dir / 'manifest.json'
    bank_manifest = json.loads(bank_manifest_path.read_text())
    store = DiskStore(bank_dir / 'bank.sqlite')
    verified = publish_offline_generation(store, identity=bank_manifest['identity'],
        namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
        spaces=tuple(bank_manifest['spaces']), shard_ids=tuple(bank_manifest['shards']),
        source_count=bank_manifest['sources'], verify_only=True)
    if canonical_json(verified) != canonical_json({key: bank_manifest[key] for key in verified}):
        raise ValueError('Published bank failed byte verification')
    if bank_manifest['identity']['model'] != asdict(config.model):
        raise ValueError('Bank model architecture differs from the evaluator')
    if bank_manifest['identity']['memory'] != asdict(config.memory):
        raise ValueError('Bank writer or stored-memory transform differs')
    config.train.retrieval = 'learned'
    config.train.selected_producers_only = False  # training-only producer policy
    config.train.bank_dir = None
    config.train.bank_read_limits = []
    config.train.payload_contrast_weight = 0.0
    config.memory.neighbors = list(limits)
    config.validate()
    torch.set_num_threads(config.train.threads)
    agent = SDKBAgent(config).to(config.train.device).eval()
    checkpoint = resolve_checkpoint(run, verify=True)
    assert_bank_writer_compatible(run, checkpoint, bank_dir, bank_manifest,
                                  training_bank_dir=training_bank_dir)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    episodes = load_episodes(episodes_file)[:max_episodes]
    if not episodes:
        raise ValueError('Evaluation needs at least one episode')
    with torch.no_grad():
        report = stored_transfer_evaluation(agent, store, episodes,
            namespace=bank_manifest['namespace'], generation=bank_manifest['generation'],
            drop_supports=True)
    report['notice'] = ('Published-corpus stored-only diagnostic. Teacher NLL and complete-support '
                        'recall do not establish free-generation accuracy or agent success.')
    identity = {'run_model_sha256': file_sha256(checkpoint / 'model.safetensors'),
                'bank_manifest_sha256': file_sha256(bank_manifest_path),
                'episodes_sha256': file_sha256(episodes_file),
                'max_episodes': max_episodes, 'read_limits': limits,
                'selection': 'learned exact scan; supplied support is never used for retrieval'}
    output.mkdir()
    atomic_json(output / 'inputs.json', identity)
    atomic_json(output / 'results.json', report)
    return {key: value for key, value in report.items() if key != 'rows'} | {
        'output': str(output), 'episodes': len(episodes)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-episodes', type=int, default=16)
    parser.add_argument('--limits', nargs='+', type=int, default=[8, 4, 2, 1])
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.bank, args.episodes, args.output,
                              max_episodes=args.max_episodes,
                              limits=tuple(args.limits)), indent=2))
