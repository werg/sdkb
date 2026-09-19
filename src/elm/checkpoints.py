"""Step-consistent local checkpoints, including the intentionally stale memory bank.

CURRENT is the only commit point. All files live in a new immutable directory;
model, optimizer/RNG, data identity, and SQLite snapshot are published together.
Root model/state links are convenience aliases, never the recovery authority.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import random
import shutil
import signal
import sqlite3
import threading
import uuid

import torch
from safetensors.torch import load_model, save_model


def _fsync(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _fsync(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def resolve_checkpoint(run: str | Path, *, verify: bool = False) -> Path:
    run = Path(run)
    if not (run / "CURRENT").exists():
        if (run / "model.safetensors").exists():
            return run  # Read compatibility with the 0.1 handoff.
        raise FileNotFoundError("No committed checkpoint")
    name = (run / "CURRENT").read_text().strip()
    if not name.startswith("step-") or Path(name).name != name:
        raise ValueError("Invalid checkpoint pointer")
    path = run / "checkpoints" / name
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["format"] != 1:
        raise ValueError("Unsupported checkpoint format")
    for filename, expected in manifest["sha256"].items():
        if Path(filename).name != filename or not (path / filename).is_file():
            raise ValueError("Incomplete checkpoint")
        if verify and _digest(path / filename) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {filename}")
    return path


def save_checkpoint(agent, optimizer, run: Path, step: int, rng: random.Random,
                    cache, fingerprint: str, *, keep: int = 2) -> Path:
    if step < 0 or keep < 1:
        raise ValueError("Invalid checkpoint step/retention")
    root = run / "checkpoints"
    root.mkdir(exist_ok=True)
    token = uuid.uuid4().hex[:12]
    pending = root / (".pending-" + token)
    pending.mkdir()
    final = root / f"step-{step:09d}-{token}"
    try:
        save_model(agent, str(pending / "model.safetensors"))
        state = {"optimizer": optimizer.state_dict(), "step": step,
                 "python_rng": rng.getstate(), "global_python_rng": random.getstate(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        torch.save(state, pending / "training_state.pt")
        # backup() sees a consistent SQLite snapshot, including committed WAL data.
        with cache.connect() as source, sqlite3.connect(pending / "training_cache.sqlite") as dest:
            source.backup(dest)
        (pending / "config.json").write_text(json.dumps(asdict(agent.config), indent=2) + "\n")
        files = [p for p in pending.iterdir() if p.is_file()]
        for path in files:
            _fsync(path)
        manifest = {"format": 1, "step": step, "dataset_sha256": fingerprint,
                    "resolved_model_revision": agent.resolved_revision,
                    "sha256": {p.name: _digest(p) for p in files}}
        (pending / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        _fsync(pending / "manifest.json")
        _fsync_dir(pending)
        os.replace(pending, final)
        _fsync_dir(root)
        _atomic_text(run / "CURRENT", final.name + "\n")
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    # Convenient legacy filenames; readers in this version always resolve CURRENT.
    for name in ("model.safetensors", "training_state.pt"):
        alias = run / (name + ".link-tmp")
        alias.unlink(missing_ok=True)
        alias.symlink_to(Path("checkpoints") / final.name / name)
        os.replace(alias, run / name)
    committed = sorted((p for p in root.glob("step-*") if p.is_dir()),
                       key=lambda p: (json.loads((p / "manifest.json").read_text())["step"],
                                      p.stat().st_mtime_ns), reverse=True)
    for old in committed[keep:]:
        if old != final:
            shutil.rmtree(old)
    return final


def restore_checkpoint(agent, optimizer, run: Path, rng: random.Random,
                       fingerprint: str) -> int:
    path = resolve_checkpoint(run, verify=True)
    if path != run:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["dataset_sha256"] != fingerprint:
            raise ValueError("Episode contents changed since checkpoint")
        if manifest["resolved_model_revision"] != agent.resolved_revision:
            raise ValueError("Base model revision changed; use the pinned original revision")
    load_model(agent, str(path / "model.safetensors"), device=agent.config.train.device)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    rng.setstate(state["python_rng"])
    random.setstate(state.get("global_python_rng", state["python_rng"]))
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if path != run:
        # Discard uncheckpointed writes. Stale/live training would otherwise change.
        destination = run / "training_cache.sqlite"
        for suffix in ("-wal", "-shm"):
            Path(str(destination) + suffix).unlink(missing_ok=True)
        temp = run / "cache-restore.tmp"
        shutil.copyfile(path / "training_cache.sqlite", temp)
        os.replace(temp, destination)
    # A crash may leave later rows or an incomplete JSON line in the log.
    log = run / "metrics.jsonl"
    if log.exists():
        rows = []
        for line in log.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["step"] <= state["step"]:
                rows.append(json.dumps(row))
        _atomic_text(log, "".join(row + "\n" for row in rows))
    return int(state["step"])


@contextmanager
def stop_on_signal():
    """Request a checkpoint at the next complete optimizer step, not mid-backward."""
    requested = {"signal": None}
    old = {}
    if threading.current_thread() is threading.main_thread():
        def handle(signum, _frame):
            requested["signal"] = signum
        for sig in (signal.SIGINT, signal.SIGTERM):
            old[sig] = signal.signal(sig, handle)
    try:
        yield requested
    finally:
        for sig, previous in old.items():
            signal.signal(sig, previous)
