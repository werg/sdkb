"""Train spatial reads against a base snapshot plus checkpointed mutable revisions."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time

from safetensors.torch import load_model
import torch

from sdkb.agent import SDKBAgent
from sdkb.bank_replay import BankWriterReplay
from sdkb.checkpoints import (resolve_checkpoint, restore_checkpoint, save_checkpoint,
                              stop_on_signal)
from sdkb.config import load_config
from sdkb.key_index import PublishedKeyIndex
from sdkb.document_ingestion import (grouped_ingestion_prefixes,
                                     source_ingestion_groups, writer_prefix_ids)
from sdkb.offline_bank import (canonical_json, publish_offline_generation,
                               stored_memory_identity)
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.optimizers import make_optimizer, optimizer_report
from sdkb.runtime import reclaim_cuda_cache
from sdkb.spatial_data import SpatialTrajectoryIndex
from sdkb.spatial_training import spatial_bank_forward, spatial_bank_pipeline_forward
from sdkb.routing_curriculum import RoutingCandidateIndex
from sdkb.store import DiskStore
from sdkb.training_bank import TrainingBank
from sdkb.tracking import Tracking
from sdkb.training import EpisodeSampler, autocast_context, environment_report, resource_report
from sdkb.trajectories import file_sha256


def _fingerprint(data: SpatialTrajectoryIndex, manifest_path: Path, settings: dict) -> str:
    payload = ":".join((data.sha256, file_sha256(manifest_path), canonical_json(settings)))
    return hashlib.sha256(payload.encode()).hexdigest()


def train(config_path: Path, data_path: Path, bank_dir: Path, output: Path,
          init_from: Path | None, *, resume: bool, steps: int, batch_size: int,
          loops: int, limits: tuple[int, ...], routing_candidates: int,
          checkpoint_every: int, train_recurrent_core: bool = False,
          microbatch_size: int | None = None, inflight: int = 1,
          sources_path: Path | None = None,
          max_unused_cuda_gib: float = 4.0,
          gradient_checkpointing: bool = False,
          profile_steps: int = 0,
          retain_writer_replay_activations: bool = False,
          cache_reclaim_host_reserve_gib: float = 16.0,
          maintenance_records_per_step: int = 0,
          routing_episodes_path: Path | None = None,
          routing_weight: float | None = None,
          inherit_bank: bool = False,
          bank_journal_path: Path | None = None,
          writer_key_learning_rate: float | None = None,
          key_stability_weight: float | None = None,
          routing_logit_scale: float | None = None,
          routing_live_weight: float | None = None,
          routing_hard_ramp_steps: int = 3000) -> dict:
    if (steps < 1 or batch_size < 1 or loops < 2 or checkpoint_every < 1
            or max_unused_cuda_gib < 0 or profile_steps < 0
            or cache_reclaim_host_reserve_gib < 0
            or maintenance_records_per_step < 0 or routing_hard_ramp_steps < 1):
        raise ValueError("Invalid spatial training schedule")
    max_unused_cuda_bytes = int(max_unused_cuda_gib * 1024 ** 3)
    cache_reclaim_host_reserve_bytes = int(cache_reclaim_host_reserve_gib * 1024 ** 3)
    pipeline = microbatch_size is not None or inflight != 1
    if pipeline:
        microbatch_size = microbatch_size or 1
        if microbatch_size < 1 or microbatch_size >= batch_size or inflight < 2:
            raise ValueError(
                "Pipelining needs a microbatch smaller than the optimizer batch "
                "and at least two batches in flight")
    config = deepcopy(load_config(config_path))
    if gradient_checkpointing:
        config.model.gradient_checkpointing = True
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
    if routing_weight is not None:
        if routing_weight <= 0:
            raise ValueError('Routing weight must be positive')
        config.train.routing_weight = routing_weight
    if writer_key_learning_rate is not None:
        config.train.writer_key_learning_rate = writer_key_learning_rate
    if key_stability_weight is not None:
        config.train.key_stability_weight = key_stability_weight
    if routing_logit_scale is not None:
        config.train.routing_logit_scale = routing_logit_scale
    if routing_live_weight is not None:
        config.train.routing_live_weight = routing_live_weight
    if routing_episodes_path is not None and (sources_path is None or not pipeline):
        raise ValueError('Routing curriculum requires source metadata and pipeline')
    if inherit_bank and (init_from is None or resume):
        raise ValueError('Bank journal inheritance needs a warm-start parent')
    if bank_journal_path is not None and (init_from is None or resume or inherit_bank):
        raise ValueError('A staged bank refresh needs a fresh warm-start output')
    # Retain enough recovery points for a long unattended trajectory stage.
    config.train.keep_checkpoints = max(config.train.keep_checkpoints, 16)
    config.train.archive_dir = None
    config.train.wandb_group = f"trajectory-spatial-r{loops}"
    config.validate()
    data = SpatialTrajectoryIndex(data_path)
    if routing_episodes_path is not None:
        trajectory_manifest = data_path.with_suffix(data_path.suffix + '.manifest.json')
        packed = json.loads(trajectory_manifest.read_text())
        if (packed['source_sha256'] != file_sha256(routing_episodes_path)
                or packed['output_sha256'] != data.sha256):
            raise ValueError('Routing query text differs from packed causal trajectories')
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
    bank_memory = stored_memory_identity(dict(bank_manifest["identity"]["memory"]))
    current_memory = stored_memory_identity(asdict(config.memory))
    if (tuple(bank_manifest["spaces"]) != tuple(f"s{i}" for i in range(len(limits)))
            or bank_memory != current_memory):
        raise ValueError("Spatial model and published bank interfaces differ")
    if config.train.writer_replay_records_per_site and (not pipeline or sources_path is None):
        raise ValueError('Writer replay requires --sources and the overlapped pipeline')
    if maintenance_records_per_step and not config.train.writer_replay_records_per_site:
        raise ValueError('Bank maintenance requires writer replay and source inputs')
    source_sha = None
    if sources_path is not None:
        source_sha = file_sha256(sources_path)
        if source_sha != bank_manifest['identity']['source_manifest_sha256']:
            raise ValueError('Writer replay source manifest differs from bank creation')
    settings = {"format": 2, "loops": loops, "limits": limits,
                "routing_candidates": routing_candidates, "batch_size": batch_size,
                "steps": steps, "sampler": "deterministic_shuffled_passes",
                "train_recurrent_core": train_recurrent_core,
                "source_manifest_sha256": source_sha}
    if pipeline:
        settings["pipeline"] = {
            "microbatch_size": microbatch_size, "inflight": inflight,
            "ordering": "deterministic_round_robin",
        }
    if maintenance_records_per_step:
        settings['maintenance_records_per_step'] = maintenance_records_per_step
    if routing_episodes_path is not None:
        settings['routing_curriculum'] = {
            'episodes_sha256': file_sha256(routing_episodes_path),
            'routing_weight': config.train.routing_weight,
            'sampled_negatives': 32,
            'hard_ramp_steps': routing_hard_ramp_steps,
            'lexical_auxiliary_fraction': .05,
        }
    if config.train.key_stability_weight:
        settings['key_stability_weight'] = config.train.key_stability_weight
    if config.train.routing_logit_scale != 1.0:
        settings['routing_logit_scale'] = config.train.routing_logit_scale
    if config.train.routing_live_weight:
        settings['routing_live_weight'] = config.train.routing_live_weight
    if config.train.writer_key_learning_rate is not None:
        settings['writer_key_learning_rate'] = config.train.writer_key_learning_rate
    refresh_manifest = None
    if bank_journal_path is not None:
        refresh_manifest_path = bank_journal_path.parent / 'manifest.json'
        refresh_manifest = json.loads(refresh_manifest_path.read_text())
        if not refresh_manifest.get('complete'):
            raise ValueError('Staged bank refresh is incomplete')
        settings['bank_refresh_manifest_sha256'] = file_sha256(refresh_manifest_path)
    if resume and (output / 'spatial-inputs.json').exists():
        prior = json.loads((output / 'spatial-inputs.json').read_text())
        inherited = prior.get('inherit_bank_from')
        if inherited is not None:
            settings['inherit_bank_from'] = inherited
        refreshed = prior.get('bank_refresh_manifest_sha256')
        if refreshed is not None:
            settings['bank_refresh_manifest_sha256'] = refreshed
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
            parameter.requires_grad_(bool(config.train.writer_replay_records_per_site))
    agent.train()
    initialization = None
    if not resume:
        checkpoint = resolve_checkpoint(init_from, verify=True)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if manifest["resolved_model_revision"] != agent.resolved_revision:
            raise ValueError("Warm-start model revision differs")
        missing, unexpected = load_model(agent, str(checkpoint / "model.safetensors"),
                                         strict=False, device=config.train.device)
        allowed_missing = {name for name in missing if name.startswith('distance_gates.')}
        if set(missing) != allowed_missing or unexpected:
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
        if inherit_bank:
            parent_state = json.loads((checkpoint / 'bank-state.json').read_text())
            parent_cache = DiskStore(init_from / 'training_cache.sqlite')
            if parent_cache.mutable_bank_state() != parent_state:
                raise ValueError('Parent bank journal has moved beyond its checkpoint')
            with parent_cache.connect() as source, sqlite3.connect(
                    output / 'training_cache.sqlite') as target:
                source.backup(target)
            initialization['inherited_bank_state'] = parent_state
        if bank_journal_path is not None:
            if (refresh_manifest['writer_checkpoint_sha256']
                    != initialization['checkpoint_sha256']
                    or refresh_manifest['bank_manifest_sha256']
                    != file_sha256(bank_manifest_path)
                    or refresh_manifest['source_manifest_sha256'] != source_sha):
                raise ValueError('Refreshed bank and warm-start model differ')
            staged = DiskStore(bank_journal_path)
            if staged.mutable_bank_state() != refresh_manifest['journal_state']:
                raise ValueError('Refreshed bank journal differs from its manifest')
            with staged.connect() as source, sqlite3.connect(
                    output / 'training_cache.sqlite') as target:
                source.backup(target)
            initialization['bank_refresh_manifest_sha256'] = settings[
                'bank_refresh_manifest_sha256']
            initialization['inherited_bank_state'] = refresh_manifest['journal_state']
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
    training_bank = TrainingBank(store, cache, index)
    source_rows = {}
    if sources_path is not None:
        with sources_path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                source_rows[row['record_id']] = row
        if set(map(str, next(iter(index.spaces.values())).ids)) - source_rows.keys():
            raise ValueError('Writer replay manifest does not cover the published bank')
    source_groups = source_ingestion_groups(list(source_rows.values()))
    routing_teacher = (RoutingCandidateIndex(
        sources_path, ramp_steps=routing_hard_ramp_steps)
                       if routing_episodes_path is not None else None)
    queries = {}
    if routing_episodes_path is not None:
        with routing_episodes_path.open(encoding='utf-8') as handle:
            for line in handle:
                episode = json.loads(line)
                episode_id = episode['episode_id']
                if episode_id in queries and queries[episode_id] != episode['query']:
                    raise ValueError('Episode query identity changed')
                queries[episode_id] = episode['query']

    @lru_cache(maxsize=256)
    def ingestion_prefixes(document_id, parts, mode, domain):
        return grouped_ingestion_prefixes(
            document_id, parts, generation=bank_manifest['generation'],
            scope={'domain': domain}, mode=mode)

    @lru_cache(maxsize=2048)
    def writer_tokens(record_id):
        row = source_rows[record_id]
        document_id, parts, mode = source_groups[record_id]
        messages = ingestion_prefixes(
            document_id, parts, mode, row.get('domain', 'research'))[record_id]
        return tuple(writer_prefix_ids(agent.tokenizer, messages))

    class WriterInputs(dict):
        def __getitem__(self, record_id):
            return torch.tensor([writer_tokens(record_id)], dtype=torch.long,
                                device=agent.device)

        def keys(self):
            return source_rows.keys()

    writer_inputs = WriterInputs()
    sampler = EpisodeSampler(len(data), seed=config.train.seed)
    if not resume:
        save_checkpoint(agent, optimizer, output, 0, rng, cache, fingerprint,
                        keep=config.train.keep_checkpoints)
    completed, last_saved = start, 0 if not resume else None
    began = time.perf_counter()

    # The launcher acknowledges a stale stop before spawning. Never clear a new
    # request that arrives during model, index, or checkpoint initialization.
    with run_lock(output, clear_stop=False), stop_on_signal() as signal_state, \
            ExitStack() as lifecycle:
        tracker = lifecycle.enter_context(Tracking(config, output))
        environment = environment_report() | {
            "resume": resume, "attempt": tracker.attempt,
            "bank_key_index": {"kind": "resident_exact_cpu", "bytes": index.key_bytes},
            "writer_replay_ingestion_mix": {
                "holistic_complete_blob": "sha256(article-group) parity 1",
                "sequential_source_parts": "sha256(article-group) parity 0",
                "maximum_parts_per_group": 4,
            },
            "spatial": settings,
            "max_unused_cuda_gib": max_unused_cuda_gib,
            "gradient_checkpointing": config.model.gradient_checkpointing,
            "retain_writer_replay_activations": retain_writer_replay_activations,
            "cache_reclaim_host_reserve_gib": cache_reclaim_host_reserve_gib,
        }
        atomic_json(output / "environment.json", environment)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            for step in range(start, steps):
                if signal_state["signal"] is not None or stop_requested(output):
                    if last_saved != completed:
                        optimizer.zero_grad(set_to_none=True)
                        reclaim_cuda_cache(config.train.device, max_unused_cuda_bytes,
                                           force=True)
                        save_checkpoint(agent, optimizer, output, completed, rng, cache,
                                        fingerprint, keep=config.train.keep_checkpoints)
                    break
                rows = [data[sampler.index(step * batch_size + offset)]
                        for offset in range(batch_size)]
                if routing_teacher is not None:
                    for row in rows:
                        row['routing_step'] = step
                        row['routing_hard_ramp_steps'] = routing_hard_ramp_steps
                        for site in row['sites']:
                            site['routing_query_text'] = queries[site['episode_id']]
                optimizer.zero_grad(set_to_none=True)
                profiling = step < start + profile_steps
                if profiling and torch.cuda.is_available():
                    torch.cuda.synchronize(config.train.device)
                tick = time.perf_counter()
                phase_tick = tick
                phase_seconds = {}

                def finish_phase(name):
                    nonlocal phase_tick
                    if profiling and torch.cuda.is_available():
                        torch.cuda.synchronize(config.train.device)
                    now = time.perf_counter()
                    if profiling:
                        phase_seconds[f'profile_{name}_seconds'] = now - phase_tick
                    phase_tick = now

                replay = (BankWriterReplay(
                              agent, writer_inputs,
                              checkpoint_backward=not retain_writer_replay_activations)
                          if config.train.writer_replay_records_per_site else None)
                with autocast_context(config):
                    if pipeline:
                        result = spatial_bank_pipeline_forward(
                            agent, training_bank, index, rows, limits=limits,
                            routing_candidates=routing_candidates,
                            microbatch_size=microbatch_size, inflight=inflight,
                            pad_token_id=agent.tokenizer.pad_token_id or 0,
                            writer_replay=replay,
                            routing_teacher=routing_teacher,
                        )
                    else:
                        result = spatial_bank_forward(
                            agent, training_bank, index, rows, limits=limits,
                            routing_candidates=routing_candidates,
                            pad_token_id=agent.tokenizer.pad_token_id or 0,
                        )
                finish_phase('forward')
                result.loss.backward()
                finish_phase('consumer_backward')
                if replay is not None:
                    replay.backward()
                finish_phase('writer_backward')
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in agent.parameters() if parameter.requires_grad],
                    config.train.clip_grad_norm,
                )
                optimizer.step()
                finish_phase('optimizer')
                maintenance_ids = (() if replay is None else training_bank.maintenance_ids(
                    maintenance_records_per_step, exclude=replay.record_ids,
                    eligible=writer_inputs.keys()))
                refreshed_views = (replay.refresh(
                    training_bank, additional_ids=maintenance_ids,
                    optimizer_step=step + 1) if replay is not None else 0)
                finish_phase('refresh')
                loss_value = float(result.loss.detach())
                nll_value = float(result.nll.detach())
                routing_value = float(result.routing.detach())
                gradient_norm_value = float(gradient_norm)
                result_metrics = result.metrics
                replayed_records = len(set(replay.record_ids)) if replay is not None else 0
                del result, replay, gradient_norm
                optimizer.zero_grad(set_to_none=True)
                cache_metrics = reclaim_cuda_cache(
                    config.train.device, max_unused_cuda_bytes,
                    min_host_available_bytes=cache_reclaim_host_reserve_bytes)
                finish_phase('cache_reclaim')
                completed = step + 1
                elapsed = time.perf_counter() - tick
                row = {
                    "step": completed, "loss": loss_value,
                    "nll": nll_value, "routing": routing_value,
                    "gradient_norm": gradient_norm_value, "step_seconds": elapsed,
                    "tokens": sum(item["tokens"] for item in rows),
                    "tokens_per_second": sum(item["tokens"] for item in rows) / elapsed,
                    "trajectory_rows": batch_size, **result_metrics,
                    "replayed_records": replayed_records,
                    "maintenance_records": len(maintenance_ids),
                    "refreshed_record_views": refreshed_views,
                    "bank_active_views": training_bank.sizes()['views'],
                    "bank_journal_cursor": training_bank.cursor,
                    **cache_metrics, **phase_seconds,
                }
                if torch.cuda.is_available() and str(config.train.device).startswith('cuda'):
                    row.update(
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(
                            config.train.device),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(
                            config.train.device),
                    )
                if profiling or completed == 1 or completed % config.train.log_every == 0:
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    tracker.log(row)
                    print(json.dumps(row), flush=True)
                if completed % checkpoint_every == 0:
                    reclaim_cuda_cache(config.train.device, max_unused_cuda_bytes,
                                       force=True)
                    save_checkpoint(agent, optimizer, output, completed, rng, cache,
                                    fingerprint, keep=config.train.keep_checkpoints)
                    last_saved = completed
            if completed == steps and last_saved != completed:
                reclaim_cuda_cache(config.train.device, max_unused_cuda_bytes,
                                   force=True)
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
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--inflight", type=int, default=1)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--max-unused-cuda-gib", type=float, default=4.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--retain-writer-replay-activations", action="store_true")
    parser.add_argument("--cache-reclaim-host-reserve-gib", type=float, default=16.0)
    parser.add_argument("--maintenance-records-per-step", type=int, default=0)
    parser.add_argument("--routing-episodes", type=Path)
    parser.add_argument("--routing-weight", type=float)
    parser.add_argument("--inherit-bank", action="store_true")
    parser.add_argument("--bank-journal", type=Path)
    parser.add_argument("--writer-key-learning-rate", type=float)
    parser.add_argument("--key-stability-weight", type=float)
    parser.add_argument("--routing-logit-scale", type=float)
    parser.add_argument("--routing-live-weight", type=float)
    parser.add_argument("--routing-hard-ramp-steps", type=int, default=3000)
    args = parser.parse_args()
    print(json.dumps(train(
        args.config, args.data, args.bank, args.output, args.init_from,
        resume=args.resume, steps=args.steps, batch_size=args.batch_size,
        loops=args.loops, limits=tuple(args.limits),
        routing_candidates=args.routing_candidates,
        checkpoint_every=args.checkpoint_every,
        train_recurrent_core=args.train_recurrent_core,
        microbatch_size=args.microbatch_size, inflight=args.inflight,
        sources_path=args.sources,
        max_unused_cuda_gib=args.max_unused_cuda_gib,
        gradient_checkpointing=args.gradient_checkpointing,
        profile_steps=args.profile_steps,
        retain_writer_replay_activations=args.retain_writer_replay_activations,
        cache_reclaim_host_reserve_gib=args.cache_reclaim_host_reserve_gib,
        maintenance_records_per_step=args.maintenance_records_per_step,
        routing_episodes_path=args.routing_episodes,
        routing_weight=args.routing_weight,
        inherit_bank=args.inherit_bank,
        bank_journal_path=args.bank_journal,
        writer_key_learning_rate=args.writer_key_learning_rate,
        key_stability_weight=args.key_stability_weight,
        routing_logit_scale=args.routing_logit_scale,
        routing_live_weight=args.routing_live_weight,
        routing_hard_ramp_steps=args.routing_hard_ramp_steps,
    ), indent=2))
