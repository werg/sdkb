"""Resumable pretrained text-context utility; no writer, training or tool execution."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import shutil

import torch

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import stop_on_signal
from sdkb.config import load_config
from sdkb.data import evidence_ids, load_episodes
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.runtime import available_host_memory, configure_memory, compute_watchdog, memory_metrics
from sdkb.training import autocast_context, environment_report
from sdkb.trajectories import file_sha256


def summarize(rows):
    summary, pairs, groups = {}, defaultdict(dict), defaultdict(list)
    for condition in ('selected_text', 'none'):
        values = [r for r in rows if r['condition'] == condition]
        summary[condition] = dict(episodes=len(values),
            token_weighted_nll=sum(r['sequence_nll'] for r in values)/sum(r['token_count'] for r in values),
            mean_episode_nll=sum(r['mean_nll'] for r in values)/len(values))
    for row in rows:
        pairs[row['episode']][row['condition']] = row
    for pair in pairs.values():
        groups[pair['none']['repository']].append(pair['none']['mean_nll']-pair['selected_text']['mean_nll'])
    values, rng, samples = list(groups.values()), random.Random(191), []
    for _ in range(2000):
        selected = [rng.choice(values) for _ in values]
        samples.append(sum(map(sum, selected))/sum(map(len, selected)))
    samples.sort()
    summary['paired_context_benefit'] = dict(mean_episode_nll_reduction=sum(map(sum, values))/sum(map(len, values)),
        repository_groups=len(groups), repository_bootstrap_interval_95=[samples[50], samples[1949]],
        scope='Descriptive repository-cluster bootstrap, one pretrained model; not agent success or a training-seed interval.')
    return summary


@torch.no_grad()
def run(config_path, episodes_path, output):
    config = load_config(config_path)
    if config.train.arm != 'oracle_text' or config.model.loops != 1 or config.train.evidence_scope != 'available':
        raise ValueError('Readiness requires the one-pass text path and all declared prior context')
    episodes = load_episodes(episodes_path)
    if not episodes:
        raise ValueError('Nonempty episodes required')
    inputs = dict(config_sha256=file_sha256(config_path), episodes_sha256=file_sha256(episodes_path),
                  evaluator_sha256=file_sha256(__file__), model_id=config.model.model_id,
                  revision=config.model.revision, conditions=['selected_text', 'none'])
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output), stop_on_signal() as signals:
        identity = output/'inputs.json'
        if identity.exists() and json.loads(identity.read_text()) != inputs:
            raise ValueError('Text readiness inputs changed')
        atomic_json(identity, inputs)
        rows = []
        if (output/'progress.json').exists():
            prior = json.loads((output/'progress.json').read_text())
            if prior['inputs'] != inputs:
                raise ValueError('Text readiness progress inputs changed')
            rows = prior['rows']
        jobs = [(e, c) for e in episodes for c in inputs['conditions']]
        if (len(rows) > len(jobs) or any((r['episode'], r['condition']) != (e.episode_id, c)
                or r['token_count'] < 1 or not math.isfinite(r['sequence_nll'])
                for r, (e, c) in zip(rows, jobs))):
            raise ValueError('Invalid completed scoring prefix')
        def save_progress():
            if shutil.disk_usage(output).free < config.train.min_free_disk_bytes:
                raise OSError('Text readiness disk reserve reached')
            atomic_json(output/'progress.json', dict(inputs=inputs, rows=rows))
        def want_stop():
            available = available_host_memory()
            return bool(signals['signal'] or stop_requested(output) or (
                available is not None and available < config.train.min_system_available_bytes))
        if want_stop():
            save_progress()
            return
        if len(rows) < len(jobs):
            configure_memory(config.train)
            torch.set_num_threads(config.train.threads)
            torch.manual_seed(config.train.seed)
            agent = SDKBAgent(config).to(config.train.device).eval().requires_grad_(False)
            if agent.resolved_revision != config.model.revision and config.model.backend == 'hf':
                raise ValueError('Resolved base model differs from pinned revision')
            atomic_json(output/'environment.json', environment_report())
            for episode, condition in jobs[len(rows):]:
                if want_stop():
                    save_progress()
                    return
                visible = evidence_ids(episode, 'available')
                evidence = '\n'.join(s.text for s in episode.supports if s.record_id in visible)
                prompt = agent.prompt_ids(episode.query, evidence if condition == 'selected_text' else '')
                target = agent.target_ids(episode.answer)
                with compute_watchdog(config.train.stall_timeout_seconds, device=config.train.device), autocast_context(config):
                    nll = float(agent.conditioned_nll(prompt, target, None, loops=1, reduction='sum'))
                if not math.isfinite(nll):
                    raise FloatingPointError('Nonfinite readiness score')
                rows.append(dict(episode=episode.episode_id, trajectory=episode.environment,
                    repository=episode.provenance.get('split_group', episode.environment), condition=condition,
                    prompt_tokens=prompt.numel(), token_count=target.numel(), sequence_nll=nll,
                    mean_nll=nll/target.numel(), support_annotation=episode.support_annotation))
                save_progress()
                if len(rows) % 16 == 0:
                    print(json.dumps(dict(completed=len(rows), total=len(jobs), memory=memory_metrics(config.train.device))), flush=True)
        atomic_json(output/'results.json', dict(inputs=inputs, rows=rows, summary=summarize(rows),
            notice='Pretrained one-pass teacher-forced text-context utility. No memory writer, model training, source-command execution or measured agent success. Text context changes token/compute budgets.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.episodes, args.output)
