"""Diagnose identifier fit on training examples; never treat this as held-out evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate_stored_generation import evaluate
from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes, save_episodes
from sdkb.evaluation import build_shared_bank
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.store import DiskStore
from sdkb.training import autocast_context, config_from_run
from sdkb.trajectories import file_sha256
from safetensors.torch import load_model
import torch


def diagnose(study, output):
    inputs = json.loads((study / 'inputs.json').read_text())
    config = config_from_run(Path(inputs['source_checkpoint']))
    episodes = load_episodes(config.train.episodes_file)
    if file_sha256(config.train.episodes_file) != inputs['episodes_sha256']:
        raise ValueError('Training data identity differs')
    worlds = list(dict.fromkeys(e.environment for e in episodes))[:8]
    episodes = [e for e in episodes if e.environment in worlds and e.task_family == 'multiuse/identifier']
    if len(episodes) != 16:
        raise ValueError('Expected two identifier questions per training world')
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'episodes.jsonl'
    save_episodes(path, episodes)
    with run_lock(output):
        for name, record in inputs['configs'].items():
            if stop_requested(output):
                raise RuntimeError('Diagnostic stopped')
            checkpoint = resolve_checkpoint(Path(record['run']), verify=True)
            config = config_from_run(checkpoint)
            torch.set_num_threads(config.train.threads)
            torch.manual_seed(config.train.seed)
            agent = SDKBAgent(config).to(config.train.device).eval()
            load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
            bank = output / f'{name}.sqlite'
            with torch.no_grad(), autocast_context(config):
                build_shared_bank(agent, DiskStore(bank), episodes)
            del agent
            report = evaluate(checkpoint, bank, path, 8, 24)
            report['notice'] = ('TRAINING EXAMPLES: first eight training worlds, identifier questions only. '
                                'This measures fit and memory dependence, not generalization.')
            report['training_dataset_sha256'] = inputs['episodes_sha256']
            atomic_json(output / f'{name}.json', report)
            print(json.dumps({'arm': name, 'by_family': report['by_family']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.study.resolve(), args.output.resolve())
