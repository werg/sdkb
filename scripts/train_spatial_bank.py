"""Train whole-trajectory spatial reads against one immutable published bank."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from safetensors.torch import load_model
import torch

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import (resolve_checkpoint, restore_checkpoint, save_checkpoint,
                              stop_on_signal)
from sdkb.config import load_config
from sdkb.key_index import PublishedKeyIndex
from sdkb.offline_bank import canonical_json, publish_offline_generation
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.optimizers import make_optimizer, optimizer_report
from sdkb.spatial_data import SpatialTrajectoryIndex
from sdkb.spatial_training import spatial_bank_forward
from sdkb.store import DiskStore
from sdkb.tracking import Tracking
from sdkb.training import EpisodeSampler, autocast_context, environment_report, resource_report
from sdkb.trajectories import file_sha256


def _fingerprint(data: SpatialTrajectoryIndex, manifest_path: Path, settings: dict) -> str:
    payload = ":".join((data.sha256, file_sha256(manifest_path), canonical_json(settings)))
    return hashlib.sha256(payload.encode()).hexdigest()


def train(config_path: Path, data_path: Path, bank_dir: Path, output: Path,
          init_from: Path | None, *, resume: bool, steps: int, batch_size: int,
          loops: int, limits: tuple[int, ...], routing_candidates: int,
          checkpoint_every: int, train_recurrent_core: bool = False) -> dict:
    if steps < 1 or batch_size < 1 or loops < 2 or checkpoint_every < 1:
        raise ValueError("Invalid spatial training schedule")
    config = deepcopy(load_config(config_path))
    config.model.loops = loops
    if train_recurrent_core:
        config.model.freeze_backbone = False
        config.model.backbone_train_scope = "recurrent_core"
    config.memory.read_steps = loops - 1
    config.train.steps = steps
    config.train.batch_size = batch_size
    config.train.gradient_accumulation = 1
    config.train.bank_dir = None
    config.train.episodes_file = None
    config.train.tokenized_episodes_file = None
    config.train.live_fraction = 0
    config.train.checkpoint_every = checkpoint_every
    # This stage normally has only initial and final checkpoints. A high retention
    # count avoids deleting directories while the unattended handoff is active.
    config.train.keep_checkpoints = max(config.train.keep_checkpoints, 16)
    config.train.archive_dir = None
    config.train.wandb_group = f"trajectory-spatial-r{loops}"
    config.validate()
    data = SpatialTrajectoryIndex(data_path)
    bank_manifest_path = bank_dir / "manifest.json"
    bank_manifest = json.loads(bank_manifest_path.read_text())
    store = DiskStore(bank_dir / "bank.sqlite")
    verified = publish_offline_generation(
        store, identity=bank_manifest["identity"], namespace=bank_manifest["namespace"],
        generation=bank_manifest["generation"], spaces=tuple(bank_manifest["spaces"]),
        shard_ids=tuple(bank_manifest["shards"]), source_count=bank_manifest["sources"],
        verify_only=True,
    )
    if canonical_json(verified) != canonical_json({key: bank_manifest[key] for key in verified}):
        raise ValueError("Published bank differs from its verified manifest")
    bank_memory = dict(bank_manifest["identity"]["memory"])
    current_memory = asdict(config.memory)
    # Recurrent read depth is a consumer schedule, not part of stored key/payload
    # encoding. Every other memory field remains pinned to the bank generation.
    bank_memory.pop("read_steps", None)
    current_memory.pop("read_steps", None)
    if (tuple(bank_manifest["spaces"]) != tuple(f"s{i}" for i in range(len(limits)))
            or bank_memory != current_memory):
        raise ValueError("Spatial model and published bank interfaces differ")
    settings = {"format": 1, "loops": loops, "limits": limits,
                "routing_candidates": routing_candidates, "batch_size": batch_size,
                "steps": steps, "sampler": "deterministic_shuffled_passes",
                "train_recurrent_core": train_recurrent_core}
    fingerprint = _fingerprint(data, bank_manifest_path, settings)

    if resume:
        if init_from is not None:
            raise ValueError("Choose resume or warm start")
        old = json.loads((output / "spatial-inputs.json").read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError("Spatial data, bank, or scientific schedule changed")
    else:
        if init_from is None:
            raise ValueError("A spatial stage needs an explicit compatible warm start")
        output.mkdir(parents=True, exist_ok=False)

    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    torch.set_num_threads(config.train.threads)
    agent = SDKBAgent(config).to(config.train.device)
    for name, parameter in agent.named_parameters():
        if name.startswith(("write_slots", "key_head.", "value_head.",
                            "address_maps.", "codecs.")):
            parameter.requires_grad_(False)
    agent.train()
    initialization = None
    if not resume:
        checkpoint = resolve_checkpoint(init_from, verify=True)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if manifest["resolved_model_revision"] != agent.resolved_revision:
            raise ValueError("Warm-start model revision differs")
        missing, unexpected = load_model(agent, str(checkpoint / "model.safetensors"),
                                         strict=False, device=config.train.device)
        if missing or unexpected:
            raise ValueError(f"Spatial warm start is incompatible: {missing=}, {unexpected=}")
        if loops == 2 and file_sha256(checkpoint / "model.safetensors") != \
                bank_manifest["identity"]["writer_checkpoint_sha256"]:
            raise ValueError("Initial spatial stage must start from the bank writer snapshot")
        if loops > 2:
            parent_inputs = init_from / "spatial-inputs.json"
            if (not parent_inputs.is_file()
                    or json.loads(parent_inputs.read_text())["bank_manifest_sha256"]
                    != file_sha256(bank_manifest_path)):
                raise ValueError("Deeper spatial stages require a parent trained on this bank")
        initialization = {"checkpoint": str(checkpoint),
                          "checkpoint_sha256": file_sha256(checkpoint / "model.safetensors"),
                          "optimizer_reset": True, "bank_writer_exact": loops == 2}
        atomic_json(output / "initialization.json", initialization)
    atomic_json(output / "spatial-inputs.json", settings | {
        "fingerprint": fingerprint, "data": str(data_path), "data_sha256": data.sha256,
        "bank": str(bank_dir), "bank_manifest_sha256": file_sha256(bank_manifest_path),
        "generation": bank_manifest["generation"], "rows": len(data),
    })
    atomic_json(output / "config.json", asdict(config))
    optimizer = make_optimizer(agent)
    rng = random.Random(config.train.seed)
    cache = DiskStore(output / "training_cache.sqlite")
    start = restore_checkpoint(agent, optimizer, output, rng, fingerprint) if resume else 0
    if start > steps:
        raise ValueError("Saved spatial step exceeds requested budget")
    atomic_json(output / "optimizer.json", {
        "kind": config.train.optimizer, "groups": optimizer_report(optimizer),
        "resumed": resume, "completed_steps": start,
    })
    index = PublishedKeyIndex(
        store, namespace=bank_manifest["namespace"], generation=bank_manifest["generation"],
        spaces=tuple(bank_manifest["spaces"]), expected_sources=bank_manifest["sources"],
    )
    sampler = EpisodeSampler(len(data), seed=config.train.seed)
    if not resume:
        save_checkpoint(agent, optimizer, output, 0, rng, cache, fingerprint,
                        keep=config.train.keep_checkpoints)
    completed, last_saved = start, 0 if not resume else None
    began = time.perf_counter()

    with run_lock(output), stop_on_signal() as signal_state, ExitStack() as lifecycle:
        tracker = lifecycle.enter_context(Tracking(config, output))
        environment = environment_report() | {
            "resume": resume, "attempt": tracker.attempt,
            "bank_key_index": {"kind": "resident_exact_cpu", "bytes": index.key_bytes},
            "spatial": settings,
        }
        atomic_json(output / "environment.json", environment)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            for step in range(start, steps):
                if signal_state["signal"] is not None or stop_requested(output):
                    if last_saved != completed:
                        save_checkpoint(agent, optimizer, output, completed, rng, cache,
                                        fingerprint, keep=config.train.keep_checkpoints)
                    break
                rows = [data[sampler.index(step * batch_size + offset)]
                        for offset in range(batch_size)]
                optimizer.zero_grad(set_to_none=True)
                tick = time.perf_counter()
                with autocast_context(config):
                    result = spatial_bank_forward(
                        agent, store, index, rows, limits=limits,
                        routing_candidates=routing_candidates,
                        pad_token_id=agent.tokenizer.pad_token_id or 0,
                    )
                result.loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in agent.parameters() if parameter.requires_grad],
                    config.train.clip_grad_norm,
                )
                optimizer.step()
                completed = step + 1
                elapsed = time.perf_counter() - tick
                row = {
                    "step": completed, "loss": float(result.loss.detach()),
                    "nll": float(result.nll.detach()),
                    "routing": float(result.routing.detach()),
                    "gradient_norm": float(gradient_norm), "step_seconds": elapsed,
                    "tokens": sum(item["tokens"] for item in rows),
                    "tokens_per_second": sum(item["tokens"] for item in rows) / elapsed,
                    "trajectory_rows": batch_size, **result.metrics,
                }
                if completed == 1 or completed % config.train.log_every == 0:
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    tracker.log(row)
                    print(json.dumps(row), flush=True)
                if completed % checkpoint_every == 0:
                    save_checkpoint(agent, optimizer, output, completed, rng, cache,
                                    fingerprint, keep=config.train.keep_checkpoints)
                    last_saved = completed
            if completed == steps and last_saved != completed:
                save_checkpoint(agent, optimizer, output, completed, rng, cache,
                                fingerprint, keep=config.train.keep_checkpoints)
                last_saved = completed
    summary = {
        "completed_steps": completed, "requested_steps": steps,
        "complete": completed == steps, "elapsed_seconds": time.perf_counter() - began,
        "examples_seen": completed * batch_size, "sites_seen": completed * batch_size
                      * len(data[0]["sites"]), "resources": resource_report(),
    }
    atomic_json(output / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--limits", nargs="+", type=int, default=[16, 8, 4, 4])
    parser.add_argument("--routing-candidates", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--train-recurrent-core", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(
        args.config, args.data, args.bank, args.output, args.init_from,
        resume=args.resume, steps=args.steps, batch_size=args.batch_size,
        loops=args.loops, limits=tuple(args.limits),
        routing_candidates=args.routing_candidates,
        checkpoint_every=args.checkpoint_every,
        train_recurrent_core=args.train_recurrent_core,
    ), indent=2))
