"""Reproduce the pilot's backward failure from an isolated checkpoint copy.

No checkpoint writes, tracking, or archive retention. This deliberately supports
only the pilot's all-live oracle-memory recipe. Successful steps must match its
training log before drawing conclusions about the failure.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import json
import random

import torch

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import restore_checkpoint
from sdkb.episode_index import EpisodeIndex
from sdkb.replay import ReplayTape
from sdkb.readers import Statistics
from sdkb.training import autocast_context, config_from_run, stored_channel


def diagnose(run, steps, anomaly):
    config = config_from_run(run)
    assert config.train.arm == 'memory' and config.train.retrieval == 'oracle'
    assert config.train.live_fraction == 1 and config.memory.compaction == 'none'
    torch.set_num_threads(config.train.threads)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    rng = random.Random(config.train.seed)
    agent = SDKBAgent(config).to(config.train.device).train()
    base_ids = {id(p) for p in agent.backbone.base.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': [p for p in agent.parameters() if p.requires_grad and id(p) in base_ids],
         'lr': config.train.backbone_learning_rate},
        {'params': [p for p in agent.parameters() if p.requires_grad and id(p) not in base_ids],
         'lr': config.train.learning_rate}])
    episodes = EpisodeIndex(config.train.episodes_file)
    start = restore_checkpoint(agent, optimizer, run, rng, episodes.sha256)
    original_mean = Statistics.mean
    def traced_mean(stats):
        if (stats.mass.detach() < 1e-12).any():
            print(json.dumps(dict(small_mass=stats.mass.detach().flatten().tolist(),
                numerator_absmax=stats.numerator.detach().abs().amax(-1).flatten().tolist(),
                log_scale=stats.log_scale.detach().flatten().tolist())), flush=True)
        return original_mean(stats)
    Statistics.mean = traced_mean
    for step in range(start, start + steps):
        with torch.autograd.detect_anomaly() if anomaly and step == start + steps - 1 else nullcontext():
            agent.backbone.loops = rng.choice(config.train.loop_counts)
            optimizer.zero_grad(set_to_none=True)
            losses, episode_ids = [], []
            for micro in range(config.train.gradient_accumulation):
                episode = rng.choice(episodes)
                episode_ids.append(episode.episode_id)
                print(json.dumps(dict(step=step+1, micro=micro, episode=episode.episode_id,
                                      loops=agent.backbone.loops)), flush=True)
                tape = ReplayTape(verify_outputs=config.train.verify_replay)
                required = [i for i, s in enumerate(episode.supports) if s.record_id in episode.required_ids]
                with autocast_context(config):
                    records = []
                    for source in episode.supports:
                        ids = agent.text_ids(source.text, source=True)
                        assert rng.random() < config.train.live_fraction
                        def producer(ids=ids):
                            return stored_channel(agent, agent.produce(ids))
                        records.append(tape.capture(agent, producer) if config.train.replay else producer())
                    prompt = agent.prompt_ids(episode.query)
                    target = agent.target_ids(episode.answer)
                    result = agent(prompt, target, records, required, step=step, compact=False)
                    loss = result.loss
                    if config.train.oracle_anchor_weight:
                        support = '\n'.join(s.text for s in episode.supports if s.record_id in episode.required_ids)
                        anchor = agent.conditioned_nll(agent.prompt_ids(episode.query, support), target, None,
                                                       loops=config.train.oracle_anchor_loops)
                        loss = loss + config.train.oracle_anchor_weight * anchor
                    loss = loss / config.train.gradient_accumulation
                losses.append(float(loss.detach()))
                loss.backward()
                if config.train.replay:
                    tape.backward()
                bad = [n for n, p in agent.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad:
                    print(json.dumps(dict(step=step+1, micro=micro, nonfinite_parameters=bad)), flush=True)
                    raise FloatingPointError('Nonfinite gradients; see parameter report')
            norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), config.train.clip_grad_norm,
                                                 error_if_nonfinite=True)
            optimizer.step()
            print(json.dumps(dict(completed=step+1, optimization_loss=sum(losses),
                                  grad_norm=float(norm), episodes=episode_ids)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--anomaly', action='store_true')
    args = parser.parse_args()
    diagnose(args.run, args.steps, args.anomaly)
