"""Read-only bridge-gate sensitivity using one frozen stored bank and read plans."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

import torch
from safetensors.torch import load_model

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import evidence_ids, load_episodes
from sdkb.operations import atomic_json, run_lock
from sdkb.runtime import configure_memory
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import autocast_context, config_from_run
from sdkb.trajectories import file_sha256


POLICIES = ('trained', 'memory_0.30', 'update_0.30', 'both_0.30', 'both_0.50')


def _logit(probability):
    return math.log(probability / (1 - probability))


def _paired(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['trajectory']].append(
            (row['zero_values_nll'] - row['real_nll']) / row['target_tokens'])
    values = list(groups.values())
    rng = random.Random(233)
    samples = []
    for _ in range(2000):
        selected = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, selected)) / sum(map(len, selected)))
    samples.sort()
    return dict(mean_episode_nll_benefit=sum(map(sum, values)) / sum(map(len, values)),
                trajectory_groups=len(values), trajectory_bootstrap_interval_95=[samples[50], samples[1949]])


@torch.no_grad()
def run(source: Path, episodes_file: Path, bank: Path, baseline_report: Path, output: Path) -> dict:
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(source)
    if (config.train.arm != 'memory' or config.model.loops != 2
            or config.memory.read_steps != 1 or config.memory.read_timing != 'loop_boundary'):
        raise ValueError('Gate sweep requires the one-read R=2 trained student')
    bank_identity = json.loads((bank.parent / 'inputs.json').read_text())
    if (bank_identity['checkpoint_model_sha256'] != file_sha256(checkpoint / 'model.safetensors')
            or bank_identity['episodes_sha256'] != file_sha256(episodes_file)):
        raise ValueError('Frozen bank differs from checkpoint or episodes')
    baseline = json.loads(baseline_report.read_text())
    episodes = load_episodes(episodes_file)
    if len(episodes) != 119 or any(baseline['summary'][c]['episodes'] != 119
                                   for c in ('all', 'zero_values', 'none')):
        raise ValueError('Expected complete held-out baseline')
    identity = dict(source_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
                    source_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
                    episodes_sha256=file_sha256(episodes_file),
                    bank_identity_sha256=file_sha256(bank.parent / 'inputs.json'),
                    baseline_report_sha256=file_sha256(baseline_report),
                    probe_sha256=file_sha256(Path(__file__)), policies=POLICIES)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        identity_file = output / 'inputs.json'
        if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
            raise ValueError('Gate sweep identity changed')
        atomic_json(identity_file, identity)
        summary_file = output / 'summary.json'
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        progress_file = output / 'progress.json'
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        if progress and progress.get('inputs') != identity:
            raise ValueError('Gate sweep progress changed')
        rows = progress.get('rows', [])
        if len(rows) > len(episodes) or any(row['episode'] != episode.episode_id
                                            for row, episode in zip(rows, episodes, strict=False)):
            raise ValueError('Gate sweep episode order changed')
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        agent = SDKBAgent(config).to(config.train.device).eval().requires_grad_(False)
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Gate sweep invoked writer')
        agent.produce = forbidden
        bridge = agent.backbone.bridge
        trained_memory = float(bridge.memory_logit.detach())
        trained_update = float(bridge.update_logit.detach())
        settings = {'trained': (trained_memory, trained_update),
                    'memory_0.30': (_logit(.30), trained_update),
                    'update_0.30': (trained_memory, _logit(.30)),
                    'both_0.30': (_logit(.30), _logit(.30)),
                    'both_0.50': (_logit(.50), _logit(.50))}
        store = DiskStore(bank)
        for episode in episodes[len(rows):]:
            prompt = agent.prompt_ids(episode.query)
            target = agent.target_ids(episode.answer)
            namespace = 'all/' + episode.environment
            common = dict(generation='teacher-eval-v1', query_time=episode.query_time)
            with autocast_context(config):
                real = read_session(agent, store, prompt, namespace=namespace,
                                    oracle_ids=evidence_ids(episode, config.train.evidence_scope), **common)
                zero = read_session(agent, store, prompt, namespace=namespace,
                                    fixed_plans=real.plans, ablate_values=True, **common)
                if real.selected_ids != zero.selected_ids:
                    raise ValueError('Zero-value control rerouted evidence')
                scores = {}
                for name in POLICIES:
                    bridge.memory_logit.fill_(settings[name][0])
                    bridge.update_logit.fill_(settings[name][1])
                    values = {condition: float(agent.conditioned_nll(prompt, target, memory, reduction='sum'))
                              for condition, memory in (('real', real.memory), ('zero_values', zero.memory),
                                                        ('none', None))}
                    if not all(math.isfinite(value) for value in values.values()):
                        raise FloatingPointError('Nonfinite gate-sweep score')
                    scores[name] = values
                bridge.memory_logit.fill_(trained_memory)
                bridge.update_logit.fill_(trained_update)
            rows.append(dict(episode=episode.episode_id, trajectory=episode.environment,
                             target_tokens=target.numel(), scores=scores))
            atomic_json(progress_file, dict(inputs=identity, rows=rows))
        result = {}
        for name in POLICIES:
            subset = [dict(trajectory=row['trajectory'], target_tokens=row['target_tokens'],
                           real_nll=row['scores'][name]['real'],
                           zero_values_nll=row['scores'][name]['zero_values']) for row in rows]
            count = sum(row['target_tokens'] for row in rows)
            result[name] = dict(memory_gate=float(torch.sigmoid(torch.tensor(settings[name][0]))),
                                update_gate=float(torch.sigmoid(torch.tensor(settings[name][1]))),
                                token_weighted_nll={condition: sum(row['scores'][name][condition] for row in rows)/count
                                                    for condition in ('real', 'zero_values', 'none')},
                                real_over_zero=_paired(subset))
        original = baseline['summary']
        if any(abs(result['trained']['token_weighted_nll'][current] - original[old]['token_weighted_nll']) > 1e-5
               for current, old in (('real', 'all'), ('zero_values', 'zero_values'), ('none', 'none'))):
            raise ValueError('Unmodified gate policy did not reproduce frozen baseline scores')
        summary = dict(inputs=identity, episodes=len(rows), same_serialized_bank_and_read_plans=True,
                       policies=result,
                       notice='Read-only parameter overlays on one frozen checkpoint; no retraining or saved model mutation. Gate values were inspected on this validation set, so policy comparisons are exploratory. Text and latent paths differ in context/compute. No agent success or capacity-substitution claim.')
        atomic_json(summary_file, summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.episodes, args.bank, args.baseline_report, args.output), indent=2))
