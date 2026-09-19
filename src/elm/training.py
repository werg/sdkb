"""Executable support/query training and stored-only evaluation reference."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import json
import hashlib
import os
import platform
import random
import resource
import subprocess
import time

import torch
from safetensors.torch import load_model, save_model

from .agent import MemoryAgent
from .config import Config
from .data import Episode, Source, counterfactual, make_episode, save_episodes, load_episodes
from .replay import ReplayTape
from .store import DiskStore, ReadPlan, Selection, StoredRecord, lookup_record


def autocast_context(config: Config):
    if config.train.precision == "bf16":
        return torch.autocast(config.train.device, dtype=torch.bfloat16)
    return nullcontext()


def environment_report() -> dict:
    report = {"python": platform.python_version(), "machine": platform.machine(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        report.update(gpu=torch.cuda.get_device_name(0),
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      bf16_supported=torch.cuda.is_bf16_supported(),
                      cuda_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["git_commit"] = None
    try:
        import transformers
        report["transformers"] = transformers.__version__
    except ImportError:
        report["transformers"] = None
    return report


def resource_report() -> dict:
    # ru_maxrss is KiB on Linux (the supported Spark platform).
    report = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if torch.cuda.is_available():
        report.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if Path("/proc/meminfo").exists():
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        report["system_mem_available_bytes"] = int(info["MemAvailable"].split()[0]) * 1024
    return report


def stored_channel(agent: MemoryAgent, outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    dtype = getattr(torch, agent.config.memory.storage_dtype)
    # Cast is part of the captured forward. Cached and live payloads pass through
    # the same representational precision, then return to the resident compute dtype.
    return tuple(x.float() if i % 2 == 0 else x.to(dtype).float() for i, x in enumerate(outputs))


def persist_outputs(store: DiskStore, agent: MemoryAgent, source: Source,
                    outputs: tuple[torch.Tensor, ...], namespace: str, generation: str) -> None:
    dtype = getattr(torch, agent.config.memory.storage_dtype)
    for space in range(len(agent.config.memory.payload_dims)):
        store.put(StoredRecord(source.record_id, outputs[2 * space][0],
            outputs[2 * space + 1][0].to(dtype), namespace=namespace, space=f"s{space}",
            generation=generation, created_at=source.created_at, source_id=source.record_id))


def read_cached(store: DiskStore, agent: MemoryAgent, source: Source,
                namespace: str, generation: str) -> tuple[torch.Tensor, ...]:
    outputs = []
    for space in range(len(agent.config.memory.payload_dims)):
        record = lookup_record(store, source.record_id, namespace=namespace,
                               space=f"s{space}", generation=generation)
        outputs.extend((record.key[None].to(agent.device), record.payload[None].float().to(agent.device)))
    return tuple(outputs)


def _save_checkpoint(agent: MemoryAgent, optimizer: torch.optim.Optimizer, output: Path,
                     step: int, rng: random.Random) -> None:
    temporary = output / "model.tmp.safetensors"
    save_model(agent, str(temporary))
    os.replace(temporary, output / "model.safetensors")
    state = {"optimizer": optimizer.state_dict(), "step": step,
             "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    # This is a locally generated state file. Load only with weights_only=True.
    torch.save(state, output / "training_state.tmp.pt")
    os.replace(output / "training_state.tmp.pt", output / "training_state.pt")


def train(config: Config, output: str | Path, *, resume: bool = False) -> dict:
    config.validate()
    output = Path(output)
    if config.train.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use configs/tiny_cpu.yaml for offline tests.")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "training_state.pt").exists():
        raise FileNotFoundError("Resume requires an existing local training_state.pt")
    torch.set_num_threads(config.train.threads)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    rng = random.Random(config.train.seed)
    agent = MemoryAgent(config).to(config.train.device)
    agent.train()
    base_ids = {id(p) for p in agent.backbone.base.parameters()}
    groups = [
        {"params": [p for p in agent.parameters() if p.requires_grad and id(p) in base_ids],
         "lr": config.train.backbone_learning_rate},
        {"params": [p for p in agent.parameters() if p.requires_grad and id(p) not in base_ids],
         "lr": config.train.learning_rate},
    ]
    optimizer = torch.optim.AdamW(groups)
    start = 0
    if resume:
        old_config = json.loads((output / "config.json").read_text())
        current = asdict(config)
        # Only the desired total step count may change in a resumed scientific run.
        old_config["train"]["steps"] = current["train"]["steps"]
        if old_config != current:
            raise ValueError("Resume config differs beyond total step count")
        load_model(agent, str(output / "model.safetensors"), device=config.train.device)
        state = torch.load(output / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    manifest = environment_report() | {"resolved_model_revision": agent.resolved_revision,
        "total_parameters": sum(p.numel() for p in agent.parameters()),
        "trainable_parameters": sum(p.numel() for p in agent.parameters() if p.requires_grad),
        "notice": "Prototype measurements; not evidence of capacity substitution."}
    (output / "environment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    episodes = (load_episodes(config.train.episodes_file) if config.train.episodes_file else
                [make_episode(i, split=f"train-{config.train.seed}", distractors=config.train.distractors)
                 for i in range(config.train.train_worlds)])
    fingerprint = hashlib.sha256(json.dumps([asdict(e) for e in episodes], sort_keys=True).encode()).hexdigest()
    data_manifest = output / "data_manifest.json"
    if resume and data_manifest.exists() and json.loads(data_manifest.read_text())["sha256"] != fingerprint:
        raise ValueError("Episode contents changed since checkpoint; refusing stale-cache reuse")
    data_manifest.write_text(json.dumps({"sha256": fingerprint, "episodes": len(episodes)}, indent=2) + "\n")
    save_episodes(output / "train.jsonl", episodes)
    cache = DiskStore(output / "training_cache.sqlite")
    generation = "mixed-training-v0"  # intentional stale/live training distribution, never evaluation
    history = []
    start_time = time.perf_counter()
    with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start, config.train.steps):
            optimizer.zero_grad(set_to_none=True)
            totals = {"loss": 0.0, "nll": 0.0, "routing_loss": 0.0, "compaction_loss": 0.0}
            for _ in range(config.train.gradient_accumulation):
                episode = rng.choice(episodes)
                tape = ReplayTape(verify_outputs=config.train.verify_replay)
                required = [i for i, source in enumerate(episode.supports) if source.record_id in episode.required_ids]
                with autocast_context(config):
                    records = []
                    if config.train.arm in {"memory", "direct_latent"}:
                        for source in episode.supports:
                            ids = agent.text_ids(source.text, source=True)
                            # Repeated source IDs use genuinely stored old payloads; first encounter populates the cache.
                            try:
                                cached = read_cached(cache, agent, source, "train", generation)
                            except KeyError:
                                with torch.no_grad():
                                    cached = stored_channel(agent, agent.produce(ids))
                                persist_outputs(cache, agent, source, cached, "train", generation)
                            if rng.random() < config.train.live_fraction:
                                def producer(ids=ids):
                                    return stored_channel(agent, agent.produce(ids))
                                value = tape.capture(agent, producer) if config.train.replay else producer()
                            else:
                                value = tuple(x.detach() for x in cached)
                            records.append(value)
                    support_text = "\n".join(s.text for s in episode.supports if s.record_id in episode.required_ids)
                    prompt = agent.prompt_ids(episode.query, support_text if config.train.arm == "oracle_text" else "")
                    compact = (config.train.arm == "memory" and config.memory.compaction != "none"
                               and step >= config.memory.compaction_warmup
                               and rng.random() < config.memory.compaction_probability)
                    result = agent(prompt, agent.target_ids(episode.answer), records, required,
                                   step=step, compact=compact)
                    loss = result.loss / config.train.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
                # Full source gradient accumulation happens BEFORE any optimizer update.
                if config.train.replay:
                    tape.backward()
                for name in totals:
                    totals[name] += float(getattr(result, name).detach()) / config.train.gradient_accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), config.train.clip_grad_norm,
                                                       error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step + 1, **totals, "grad_norm": float(grad_norm),
                   "elapsed_seconds": time.perf_counter() - start_time}
            log.write(json.dumps(row) + "\n")
            log.flush()
            history.append(row)
            if (step + 1) % config.train.log_every == 0 or step == start:
                print(json.dumps(row), flush=True)
    _save_checkpoint(agent, optimizer, output, config.train.steps, rng)
    summary = {"steps": config.train.steps, "last": history[-1] if history else None,
               "environment": manifest, "resources": resource_report(), "store": cache.sizes()}
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def config_from_run(path: Path) -> Config:
    from .config import ModelConfig, MemoryConfig, TrainConfig
    raw = json.loads((path / "config.json").read_text())
    return Config(ModelConfig(**raw["model"]), MemoryConfig(**raw["memory"]), TrainConfig(**raw["train"]))


@torch.no_grad()
def build_evaluation_store(agent: MemoryAgent, store: DiskStore,
                           episodes: list[Episode], generation: str) -> None:
    if agent.training:
        raise ValueError("Freeze the writer in eval mode before building an evaluation bank")
    with autocast_context(agent.config):
        for episode in episodes:
            for name, variant in (("original", episode),
                                  ("cf_restoration", counterfactual(episode, "restoration")),
                                  ("cf_permission", counterfactual(episode, "permission"))):
                for source in variant.supports:
                    outputs = stored_channel(agent, agent.produce(agent.text_ids(source.text, source=True)))
                    persist_outputs(store, agent, source, outputs, f"{name}/{episode.episode_id}", generation)


@torch.no_grad()
def stored_evaluation(agent: MemoryAgent, store: DiskStore, episodes: list[Episode],
                      generation: str, *, generate: bool = False) -> dict:
    """No writer call here. Tests explicitly replace produce() with an exception."""
    agent.eval()
    rows = []
    actions = ["STOP", "RETRY", "RESTORE_RETRY"]
    conditions = ["all", "none", "A", "B", "irrelevant", "zero_values", "cf_restoration", "cf_permission"]
    for episode in episodes:
        for condition in conditions:
            cf = condition.startswith("cf_")
            variant = counterfactual(episode, "restoration" if condition == "cf_restoration" else "permission") if cf else episode
            namespace = f"{condition if cf else 'original'}/{episode.episode_id}"
            if condition == "none":
                selected_ids = []
            elif condition == "A":
                selected_ids = [episode.required_ids[0]]
            elif condition == "B":
                selected_ids = [episode.required_ids[1]]
            elif condition == "irrelevant":
                selected_ids = [s.record_id for s in variant.supports if s.kind == "irrelevant"]
            else:
                selected_ids = list(episode.required_ids)
            with autocast_context(agent.config):
                arm = agent.config.train.arm
                text = "\n".join(s.text for s in variant.supports if s.record_id in selected_ids)
                prompt = agent.prompt_ids(episode.query, text if arm == "oracle_text" else "")
                q = agent.query(prompt)
                payloads, selected_spaces = [], []
                for space, dim in enumerate(agent.config.memory.payload_dims):
                    # Main learned-access arm uses an actual stored-key search; intervention
                    # arms use fixed IDs so they isolate evidence, not new routing choices.
                    if agent.config.train.retrieval == "learned" and condition in {"all", "zero_values"}:
                        query_key = agent.query_maps[space](q)[0]
                        plan = store.search(query_key, namespace=namespace, space=f"s{space}",
                                            generation=generation, query_time=episode.query_time,
                                            top_k=agent.config.memory.neighbors[space])
                    else:
                        plan = ReadPlan(namespace, f"s{space}", generation, "research", episode.query_time,
                                        tuple(Selection(rid, 0.0) for rid in selected_ids))
                    values = store.fetch(plan)
                    x = torch.stack(values).float().to(agent.device) if values else q.new_empty(0, dim)
                    payloads.append(x)
                    selected_spaces.append([s.record_id for s in plan.selections])
                present = any(p.shape[0] for p in payloads)
                memory = None
                if arm == "memory" and present:
                    memory, _ = agent.read_tokens(payloads, q, ablate_values=condition == "zero_values")
                elif arm == "direct_latent" and present:
                    memory = payloads[0].reshape(1, -1, agent.width)
                    if condition == "zero_values":
                        memory = torch.zeros_like(memory)
                losses = [float(agent.conditioned_nll(prompt, agent.target_ids(a), memory)) for a in actions]
                predicted = actions[min(range(len(actions)), key=lambda i: losses[i])]
                prediction = None
                if generate:
                    if arm in {"direct_latent", "oracle_text"}:
                        raise ValueError("Greedy generation helper currently supports memory/no_memory arms")
                    supplied = payloads if arm == "memory" else None
                    if condition == "zero_values" and supplied:
                        supplied = [torch.zeros_like(p) for p in supplied]
                    prediction = agent.generate_with_payloads(prompt, supplied)
            selected_set = set().union(*(set(s) for s in selected_spaces))
            rows.append({"episode": episode.episode_id, "condition": condition,
                         "answer": variant.answer, "predicted_action": predicted,
                         "choice_correct": predicted == variant.answer,
                         "target_mean_nll": losses[actions.index(variant.answer)],
                         "selected_ids": selected_spaces,
                         "complete_support": set(episode.required_ids) <= selected_set,
                         "counterfactual_should_change": variant.answer != episode.answer,
                         "generated_text": prediction,
                         "generation_exact_match": prediction == variant.answer if generate else None})
    summary = {}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        summary[condition] = {"n": len(subset),
            "choice_accuracy": sum(r["choice_correct"] for r in subset) / len(subset),
            "mean_target_nll": sum(r["target_mean_nll"] for r in subset) / len(subset),
            "complete_support_recall": sum(r["complete_support"] for r in subset) / len(subset)}
    return {"protocol": "stored-only frozen weights; action chosen by minimum per-token NLL",
            "notice": "Small synthetic evaluation; not general coding or capacity-substitution evidence.",
            "summary": summary, "rows": rows, "resources": resource_report()}


def evaluate_run(run: str | Path, *, count: int | None = None, generate: bool = False) -> dict:
    run = Path(run)
    config = config_from_run(run)
    torch.set_num_threads(config.train.threads)
    agent = MemoryAgent(config).to(config.train.device)
    load_model(agent, str(run / "model.safetensors"), device=config.train.device)
    agent.eval()
    episodes = [make_episode(i, split=f"heldout-{config.train.seed}", distractors=config.train.distractors)
                for i in range(count or config.train.eval_worlds)]
    evaluation_dir = run / ("evaluation-" + str(time.time_ns()))
    evaluation_dir.mkdir()
    save_episodes(evaluation_dir / "episodes.jsonl", episodes)
    store = DiskStore(evaluation_dir / "stored_payloads.sqlite")
    build_evaluation_store(agent, store, episodes, generation="frozen-eval-v0")
    # Close/reopen the persistent boundary before reading anything.
    store = DiskStore(evaluation_dir / "stored_payloads.sqlite")
    result = stored_evaluation(agent, store, episodes, "frozen-eval-v0", generate=generate)
    result["store"] = store.sizes()
    (evaluation_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"directory": str(evaluation_dir), "summary": result["summary"]}, indent=2))
    return result


@torch.no_grad()
def evaluate_episode_file(run: str | Path, path: str | Path) -> dict:
    """General support/query transfer likelihood; no synthetic action assumptions.

    Writer encodes each new support once into a bank. The comparison then consumes
    only reloaded payloads, with complete, missing, or zeroed support. Full coding
    verifiers and tool execution are deliberately not substituted by this metric.
    """
    run = Path(run)
    config = config_from_run(run)
    episodes = load_episodes(path)
    torch.set_num_threads(config.train.threads)
    agent = MemoryAgent(config).to(config.train.device)
    load_model(agent, str(run / "model.safetensors"), device=config.train.device)
    agent.eval()
    output = run / ("transfer-" + str(time.time_ns()))
    output.mkdir()
    store = DiskStore(output / "payloads.sqlite")
    with autocast_context(config):
        for episode in episodes:
            for source in episode.supports:
                encoded = stored_channel(agent, agent.produce(agent.text_ids(source.text, source=True)))
                persist_outputs(store, agent, source, encoded, episode.episode_id, "frozen-v0")
    store = DiskStore(output / "payloads.sqlite")
    rows = []
    with autocast_context(config):
        for episode in episodes:
            for condition in ["all", "none", "zero_values"]:
                arm = config.train.arm
                selected = list(episode.required_ids) if condition != "none" else []
                text = "\n".join(s.text for s in episode.supports if s.record_id in selected)
                prompt = agent.prompt_ids(episode.query, text if arm == "oracle_text" else "")
                q = agent.query(prompt)
                payloads, ids = [], []
                for space, dim in enumerate(config.memory.payload_dims):
                    if config.train.retrieval == "learned" and condition != "none":
                        plan = store.search(agent.query_maps[space](q)[0], namespace=episode.episode_id,
                            space=f"s{space}", generation="frozen-v0", query_time=episode.query_time,
                            top_k=config.memory.neighbors[space])
                    else:
                        plan = ReadPlan(episode.episode_id, f"s{space}", "frozen-v0", "research",
                                        episode.query_time, tuple(Selection(i, 0.) for i in selected))
                    values = store.fetch(plan)
                    payloads.append(torch.stack(values).float().to(agent.device) if values else q.new_empty(0, dim))
                    ids.extend(s.record_id for s in plan.selections)
                memory = None
                if any(x.shape[0] for x in payloads):
                    if arm == "memory":
                        memory, _ = agent.read_tokens(payloads, q, ablate_values=condition == "zero_values")
                    elif arm == "direct_latent":
                        memory = payloads[0].reshape(1, -1, agent.width)
                        if condition == "zero_values":
                            memory = torch.zeros_like(memory)
                loss = float(agent.conditioned_nll(prompt, agent.target_ids(episode.answer), memory))
                rows.append({"episode": episode.episode_id, "condition": condition,
                             "mean_target_nll": loss, "selected_ids": ids,
                             "complete_support": set(episode.required_ids) <= set(ids)})
    result = {"protocol": "frozen writer, serialize/reload, stored-only answer likelihood",
              "notice": "Token NLL is a transfer diagnostic, not coding correctness. zero_values only changes latent payloads.",
              "rows": rows, "store": store.sizes(), "resources": resource_report()}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"directory": str(output), "rows": len(rows)}))
    return result
