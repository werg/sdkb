"""Measure frozen reader-token changes from stored real, zero and wrong values."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics

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


def _corr(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else None


@torch.no_grad()
def run(source: Path, episodes_file: Path, bank: Path, scored: Path, output: Path) -> dict:
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(source)
    if (config.train.arm != 'memory' or config.model.loops != 2
            or config.memory.read_steps != 1 or config.memory.read_timing != 'loop_boundary'):
        raise ValueError('Probe requires the trained one-read R=2 latent student')
    bank_identity = json.loads((bank.parent / 'inputs.json').read_text())
    if (bank_identity['checkpoint_model_sha256'] != file_sha256(checkpoint / 'model.safetensors')
            or bank_identity['episodes_sha256'] != file_sha256(episodes_file)):
        raise ValueError('Stored bank identity differs from source checkpoint or episodes')
    report = json.loads(scored.read_text())
    by_episode = {}
    for row in report['rows']:
        by_episode.setdefault(row['episode'], {})[row['condition']] = row
    episodes = load_episodes(episodes_file)
    if len(episodes) != 119 or set(by_episode) != {episode.episode_id for episode in episodes}:
        raise ValueError('Expected the complete held-out score set')
    identity = dict(source_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
                    source_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
                    episodes_sha256=file_sha256(episodes_file),
                    bank_identity_sha256=file_sha256(bank.parent / 'inputs.json'),
                    score_sha256=file_sha256(scored), probe_sha256=file_sha256(Path(__file__)))
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        identity_file = output / 'inputs.json'
        if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
            raise ValueError('Probe inputs changed')
        atomic_json(identity_file, identity)
        result_file = output / 'summary.json'
        if result_file.exists():
            return json.loads(result_file.read_text())
        prior = output / 'progress.json'
        progress = json.loads(prior.read_text()) if prior.exists() else {}
        if progress and progress.get('inputs') != identity:
            raise ValueError('Probe progress identity changed')
        rows = progress.get('rows', [])
        if len(rows) > len(episodes) or any(row['episode'] != episode.episode_id
                                            for row, episode in zip(rows, episodes, strict=False)):
            raise ValueError('Probe progress differs from episode order')
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        agent = SDKBAgent(config).to(config.train.device).eval().requires_grad_(False)
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        agent.produce = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError('Stored-only geometry called writer'))
        store = DiskStore(bank)
        for episode in episodes[len(rows):]:
            namespace = 'all/' + episode.environment
            prompt = agent.prompt_ids(episode.query)
            selected = evidence_ids(episode, config.train.evidence_scope)
            common = dict(generation='teacher-eval-v1', query_time=episode.query_time)
            with autocast_context(config):
                real = read_session(agent, store, prompt, namespace=namespace,
                                    oracle_ids=selected, **common)
                zero = read_session(agent, store, prompt, namespace=namespace,
                                    fixed_plans=real.plans, ablate_values=True, **common)
                wrong_plans = [[replace(plan, namespace='wrong_values/' + episode.environment)
                                for plan in step] for step in real.plans]
                wrong = read_session(agent, store, prompt,
                                     namespace='wrong_values/' + episode.environment,
                                     fixed_plans=wrong_plans, **common)
            if (real.selected_ids != zero.selected_ids or real.selected_ids != wrong.selected_ids
                    or real.selected_ids != by_episode[episode.episode_id]['all']['selected_ids']):
                raise ValueError('Read plan or selected IDs changed')
            values = [session.memory.events[0].float().flatten() for session in (real, zero, wrong)]
            if any(not torch.isfinite(value).all() for value in values):
                raise FloatingPointError('Nonfinite reader token')
            r, z, w = values
            metrics = dict(real_rms=float(r.square().mean().sqrt()),
                           zero_rms=float(z.square().mean().sqrt()),
                           wrong_rms=float(w.square().mean().sqrt()),
                           real_zero_delta_rms=float((r-z).square().mean().sqrt()),
                           real_wrong_delta_rms=float((r-w).square().mean().sqrt()),
                           real_zero_cosine=float(torch.nn.functional.cosine_similarity(r, z, dim=0)),
                           real_wrong_cosine=float(torch.nn.functional.cosine_similarity(r, w, dim=0)))
            score = by_episode[episode.episode_id]
            rows.append(dict(episode=episode.episode_id, trajectory=episode.environment,
                             payload_nll_gain=score['zero_values']['mean_nll']-score['all']['mean_nll'],
                             wrong_nll_gain=score['wrong_values']['mean_nll']-score['all']['mean_nll'],
                             **metrics))
            atomic_json(prior, dict(inputs=identity, rows=rows))
        fields = ('real_rms', 'zero_rms', 'wrong_rms', 'real_zero_delta_rms',
                  'real_wrong_delta_rms', 'real_zero_cosine', 'real_wrong_cosine')
        summary = dict(inputs=identity, episodes=len(rows), trajectories=len({r['trajectory'] for r in rows}),
            means={field: statistics.mean(row[field] for row in rows) for field in fields},
            medians={field: statistics.median(row[field] for row in rows) for field in fields},
            delta_to_real_rms_ratio=statistics.mean(row['real_zero_delta_rms']/row['real_rms'] for row in rows),
            correlation_delta_rms_payload_nll_gain=_corr(
                [row['real_zero_delta_rms'] for row in rows], [row['payload_nll_gain'] for row in rows]),
            notice='Frozen stored-only reader-token geometry, not information capacity or agent success. Real/zero/wrong use the original causal read plan; reader tokens are measured before the recurrent bridge. OS cache timing is uncontrolled.')
        atomic_json(result_file, summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--scored', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.episodes, args.bank, args.scored, args.output), indent=2))
