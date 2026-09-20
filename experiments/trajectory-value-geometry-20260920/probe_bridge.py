"""Trace where a fixed stored-value perturbation shrinks in the recurrent decoder."""
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


def rms(tensor):
    return float(tensor.float().square().mean().sqrt())


@torch.no_grad()
def trace_answer(agent, prompt, target, memory):
    captured = []
    original = agent.backbone.bridge.inject
    def inject(inputs, anchor, write):
        result = original(inputs, anchor, write)
        captured.append((inputs[:, write.start:write.start + write.tokens.shape[1]].detach().float(),
                         result[:, write.start:write.start + write.tokens.shape[1]].detach().float()))
        return result
    agent.backbone.bridge.inject = inject
    try:
        context = agent._context(prompt, memory)
        embeddings = torch.cat((context, agent.backbone.embed(target[:, :-1])), 1)
        mask = torch.ones(embeddings.shape[:2], device=agent.device, dtype=torch.long)
        states = []
        final = agent.backbone.hidden(embeddings, mask, boundary=memory.callback, trace=states)
    finally:
        agent.backbone.bridge.inject = original
    if len(states) != 2 or len(captured) != 1:
        raise ValueError('Expected exactly one native read/injection boundary')
    answer_start = context.shape[1] - 1
    return dict(reader=memory.events[0].detach().float(),
                injection_base=captured[0][0], injection_result=captured[0][1],
                core1_answer=states[0][:, answer_start:].detach().float(),
                core2_answer=states[1][:, answer_start:].detach().float(),
                core2_memory=states[1][:, prompt.shape[1]:prompt.shape[1] + agent.config.memory.read_slots].detach().float(),
                final_answer=final[:, answer_start:].detach().float())


@torch.no_grad()
def run(source: Path, episodes_file: Path, bank: Path, output: Path) -> dict:
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(source)
    if (config.train.arm != 'memory' or config.model.loops != 2
            or config.memory.read_steps != 1 or config.memory.read_timing != 'loop_boundary'):
        raise ValueError('Bridge trace requires one native read at R=2')
    bank_identity = json.loads((bank.parent / 'inputs.json').read_text())
    if (bank_identity['checkpoint_model_sha256'] != file_sha256(checkpoint / 'model.safetensors')
            or bank_identity['episodes_sha256'] != file_sha256(episodes_file)):
        raise ValueError('Stored bank differs from checkpoint or episodes')
    identity = dict(source_manifest_sha256=file_sha256(checkpoint / 'manifest.json'),
                    source_model_sha256=file_sha256(checkpoint / 'model.safetensors'),
                    episodes_sha256=file_sha256(episodes_file),
                    bank_identity_sha256=file_sha256(bank.parent / 'inputs.json'),
                    probe_sha256=file_sha256(Path(__file__)))
    episodes = load_episodes(episodes_file)
    if len(episodes) != 119:
        raise ValueError('Expected 119 held-out episodes')
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        identity_file = output / 'inputs.json'
        if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
            raise ValueError('Bridge trace inputs changed')
        atomic_json(identity_file, identity)
        summary_file = output / 'summary.json'
        if summary_file.exists():
            return json.loads(summary_file.read_text())
        progress_file = output / 'progress.json'
        progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        if progress and progress.get('inputs') != identity:
            raise ValueError('Bridge trace progress changed')
        rows = progress.get('rows', [])
        if len(rows) > len(episodes) or any(row['episode'] != episode.episode_id
                                            for row, episode in zip(rows, episodes, strict=False)):
            raise ValueError('Bridge trace episode order changed')
        configure_memory(config.train)
        torch.set_num_threads(config.train.threads)
        agent = SDKBAgent(config).to(config.train.device).eval().requires_grad_(False)
        load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Bridge trace invoked writer')
        agent.produce = forbidden
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
                    raise ValueError('Zero-value control changed read IDs')
                a, b = trace_answer(agent, prompt, target, real.memory), trace_answer(agent, prompt, target, zero.memory)
            if not torch.equal(a['core1_answer'], b['core1_answer']):
                raise ValueError('Payload affected pre-read answer state')
            fields = ('reader', 'injection_result', 'core2_answer', 'core2_memory', 'final_answer')
            measures = {name + '_rms': rms(a[name]) for name in fields}
            measures.update({name + '_delta_rms': rms(a[name]-b[name]) for name in fields})
            measures['injection_base_delta_rms'] = rms(a['injection_base']-b['injection_base'])
            if not all(torch.isfinite(value).all() for value in a.values()):
                raise FloatingPointError('Nonfinite real trace')
            rows.append(dict(episode=episode.episode_id, trajectory=episode.environment, **measures))
            atomic_json(progress_file, dict(inputs=identity, rows=rows))
        numeric = [key for key in rows[0] if key.endswith('_rms')]
        summary = dict(inputs=identity, episodes=len(rows), trajectories=len({row['trajectory'] for row in rows}),
            means={key: statistics.mean(row[key] for row in rows) for key in numeric},
            mean_delta_over_real={name: statistics.mean(row[name + '_delta_rms']/row[name + '_rms'] for row in rows)
                                  for name in fields},
            bridge_memory_gate=float(agent.backbone.bridge.memory_logit.sigmoid()),
            bridge_update_gate=float(agent.backbone.bridge.update_logit.sigmoid()),
            notice='Read-only trace of the existing native model; same prefix, fixed stored read plan and target-prefix positions for real/zero. RMS across different layers is not directly a capacity measure. No cold-cache timing or agent-success claim.')
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
