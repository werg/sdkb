"""Compare output-distribution changes from stored values and source text."""
from __future__ import annotations

import argparse
import json
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


def symmetric_kl(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError('Compared answer positions or vocabularies differ')
    p, q = a.float().log_softmax(-1), b.float().log_softmax(-1)
    value = ((p.exp() * (p - q)).sum(-1) + (q.exp() * (q - p)).sum(-1)).mean() / 2
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite output divergence')
    return float(value)


@torch.no_grad()
def run(source: Path, episodes_file: Path, bank: Path, output: Path) -> dict:
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(source)
    if (config.train.arm != 'memory' or config.model.loops != 2
            or config.memory.read_steps != 1 or config.memory.read_timing != 'loop_boundary'):
        raise ValueError('Probe requires one-read R=2 latent student')
    bank_identity = json.loads((bank.parent / 'inputs.json').read_text())
    if (bank_identity['checkpoint_model_sha256'] != file_sha256(checkpoint / 'model.safetensors')
            or bank_identity['episodes_sha256'] != file_sha256(episodes_file)):
        raise ValueError('Stored bank identity differs from source checkpoint or episodes')
    identity = dict(source_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
                    source_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
                    episodes_sha256=file_sha256(episodes_file),
                    bank_identity_sha256=file_sha256(bank.parent / 'inputs.json'),
                    probe_sha256=file_sha256(Path(__file__)))
    episodes = load_episodes(episodes_file)
    if len(episodes) != 119:
        raise ValueError('Expected all 119 repository-heldout episodes')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        identity_file = output / 'inputs.json'
        if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
            raise ValueError('Output divergence inputs changed')
        atomic_json(identity_file, identity)
        summary_file = output / 'summary.json'
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        progress_file = output / 'progress.json'
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        if progress and progress.get('inputs') != identity:
            raise ValueError('Output divergence progress changed')
        rows = progress.get('rows', [])
        if len(rows) > len(episodes) or any(row['episode'] != episode.episode_id
                                            for row, episode in zip(rows, episodes, strict=False)):
            raise ValueError('Output divergence episode order changed')
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        agent = SDKBAgent(config).to(config.train.device).eval().requires_grad_(False)
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Stored-only output probe invoked writer')
        agent.produce = forbidden
        store = DiskStore(bank)
        for episode in episodes[len(rows):]:
            prompt = agent.prompt_ids(episode.query)
            target = agent.target_ids(episode.answer)
            selected = evidence_ids(episode, config.train.evidence_scope)
            evidence = '\n'.join(source.text for source in episode.supports if source.record_id in selected)
            namespace = 'all/' + episode.environment
            common = dict(generation='teacher-eval-v1', query_time=episode.query_time)
            with autocast_context(config):
                real = read_session(agent, store, prompt, namespace=namespace,
                                    oracle_ids=selected, **common)
                zero = read_session(agent, store, prompt, namespace=namespace,
                                    fixed_plans=real.plans, ablate_values=True, **common)
                if real.selected_ids != zero.selected_ids:
                    raise ValueError('Zero-value read rerouted source IDs')
                real_states = agent.conditioned_states(prompt, target, real.memory)
                zero_states = agent.conditioned_states(prompt, target, zero.memory)
                memory_state_rms = float(real_states.float().square().mean().sqrt())
                memory_state_delta_rms = float((real_states.float()-zero_states.float()).square().mean().sqrt())
                real_logits = agent.backbone.logits(real_states).float()
                zero_logits = agent.backbone.logits(zero_states).float()
                memory_kl = symmetric_kl(real_logits, zero_logits)
                del real_states, zero_states, real_logits, zero_logits
                text_states = agent.conditioned_states(agent.prompt_ids(episode.query, evidence),
                                                       target, None, loops=1)
                none_states = agent.conditioned_states(prompt, target, None, loops=1)
                text_state_rms = float(text_states.float().square().mean().sqrt())
                text_state_delta_rms = float((text_states.float()-none_states.float()).square().mean().sqrt())
                text_logits = agent.backbone.logits(text_states).float()
                none_logits = agent.backbone.logits(none_states).float()
                text_kl = symmetric_kl(text_logits, none_logits)
                del text_states, none_states, text_logits, none_logits
            rows.append(dict(episode=episode.episode_id, trajectory=episode.environment,
                             target_tokens=target.numel(), real_zero_symmetric_kl=memory_kl,
                             text_none_symmetric_kl=text_kl,
                             real_answer_state_rms=memory_state_rms,
                             real_zero_answer_state_delta_rms=memory_state_delta_rms,
                             text_answer_state_rms=text_state_rms,
                             text_none_answer_state_delta_rms=text_state_delta_rms))
            atomic_json(progress_file, dict(inputs=identity, rows=rows))
        summary = dict(inputs=identity, episodes=len(rows), trajectories=len({row['trajectory'] for row in rows}),
            mean_real_zero_symmetric_kl=statistics.mean(row['real_zero_symmetric_kl'] for row in rows),
            mean_text_none_symmetric_kl=statistics.mean(row['text_none_symmetric_kl'] for row in rows),
            median_real_zero_symmetric_kl=statistics.median(row['real_zero_symmetric_kl'] for row in rows),
            median_text_none_symmetric_kl=statistics.median(row['text_none_symmetric_kl'] for row in rows),
            mean_real_answer_state_rms=statistics.mean(row['real_answer_state_rms'] for row in rows),
            mean_real_zero_answer_state_delta_rms=statistics.mean(row['real_zero_answer_state_delta_rms'] for row in rows),
            mean_real_zero_answer_state_delta_ratio=statistics.mean(
                row['real_zero_answer_state_delta_rms']/row['real_answer_state_rms'] for row in rows),
            mean_text_answer_state_rms=statistics.mean(row['text_answer_state_rms'] for row in rows),
            mean_text_none_answer_state_delta_rms=statistics.mean(row['text_none_answer_state_delta_rms'] for row in rows),
            mean_text_none_answer_state_delta_ratio=statistics.mean(
                row['text_none_answer_state_delta_rms']/row['text_answer_state_rms'] for row in rows),
            token_weighted_real_zero_symmetric_kl=(
                sum(row['target_tokens']*row['real_zero_symmetric_kl'] for row in rows)
                / sum(row['target_tokens'] for row in rows)),
            token_weighted_text_none_symmetric_kl=(
                sum(row['target_tokens']*row['text_none_symmetric_kl'] for row in rows)
                / sum(row['target_tokens'] for row in rows)),
            notice='Teacher-forced next-token distribution sensitivity; different text and latent prompts/compute. Stored read plans and causal boundaries are fixed. No generation, agent success or information-capacity claim.')
        atomic_json(summary_file, summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.episodes, args.bank, args.output), indent=2))
