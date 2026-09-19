"""Frozen-writer, identical-bank evaluation across recurrent depths."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from safetensors.torch import load_model

from .agent import SDKBAgent
from .checkpoints import resolve_checkpoint
from .data import load_episodes
from .evaluation import build_shared_bank, stored_transfer_evaluation
from .store import DiskStore
from .training import config_from_run, resource_report
from .trajectory_eval import build_teacher_bank, stored_teacher_evaluation


def evaluate_depths(run: str | Path, episodes_file: str | Path, output: str | Path, *,
                    depths: tuple[int, ...] = (1, 2, 3, 4), protocol: str = 'transfer',
                    max_episodes: int = 32) -> dict:
    """Materialize once with writer depth one, never regenerate between depths.

    Wall time covers prefix-plan construction, I/O, and all scored conditions; it
    is not a serving-latency benchmark. R=1 intentionally cannot read between loops.
    """
    if not depths or any(d < 1 for d in depths) or len(set(depths)) != len(depths):
        raise ValueError('Depths must be distinct positive integers')
    if protocol not in {'transfer', 'teacher'} or max_episodes < 1:
        raise ValueError('Invalid evaluation protocol/limit')
    config = config_from_run(Path(run))
    if config.model.recurrence_mode != 'middle_block' or config.model.writer_loops != 1:
        raise ValueError('Depth isolation requires middle-block recurrence and fixed writer_loops=1')
    if config.train.arm != 'memory':
        raise ValueError('This sweep evaluates a trained latent-memory student')
    if any(1 < d <= config.memory.read_steps for d in depths):
        raise ValueError('A depth would truncate the scheduled read plan; use deeper passes')
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    agent = SDKBAgent(config).to(config.train.device).eval()
    checkpoint = resolve_checkpoint(run, verify=True)
    load_model(agent, str(checkpoint / 'model.safetensors'), device=config.train.device)
    episodes = load_episodes(episodes_file)[:max_episodes]
    if not episodes:
        raise ValueError('Empty evaluation set')
    if protocol == 'transfer' and any(not e.choices for e in episodes):
        raise ValueError('Transfer accuracy requires annotated choices; use --protocol teacher')
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    store = DiskStore(directory / 'bank.sqlite')
    write_phase = (build_shared_bank(agent, store, episodes) if protocol == 'transfer'
                   else build_teacher_bank(agent, store, episodes))
    def forbidden(*args, **kwargs):
        raise AssertionError('The depth sweep must not call the writer after materialization')
    agent.produce = forbidden
    reports = []
    for depth in depths:
        agent.backbone.loops = depth
        if config.train.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        report = (stored_transfer_evaluation(agent, DiskStore(store.path), episodes, drop_supports=True)
                  if protocol == 'transfer' else stored_teacher_evaluation(agent, DiskStore(store.path), episodes))
        if config.train.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        report.update(recurrence=agent.backbone.manifest(), evaluation_wall_seconds=elapsed,
                      depth_exceeds_final_training_max=depth > max(config.train.loop_counts or [config.model.loops]),
                      depth_in_final_main_objective=depth in (config.train.loop_counts or [config.model.loops]),
                      memory_enabled=depth > 1, resources=resource_report())
        (directory / f'depth-{depth}.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append({k: v for k, v in report.items() if k != 'rows'})
    result = dict(protocol=protocol, writer_depth=1, write_phase=write_phase, same_serialized_bank=True,
                  checkpoint=str(checkpoint), bank_sizes=store.sizes(), depths=reports,
                  timing_scope='All conditions, fixed-plan candidate scoring, prefix recomputation and I/O included; not serving latency.',
                  comparison='R=1 is a no-in-loop-memory control. Compare R>=2 at fixed read budget; no claim of extrapolation gains.')
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
