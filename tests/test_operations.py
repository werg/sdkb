import json
import threading
from pathlib import Path

import pytest

from sdkb.checkpoints import resolve_checkpoint
from sdkb.training import train


def test_run_lock_and_cooperative_stop(tmp_path, tiny_config):
    from sdkb.operations import run_lock, request_stop, stop_requested
    run = tmp_path / 'run'
    with run_lock(run):
        with pytest.raises(RuntimeError, match='already running'):
            with run_lock(run):
                pass
        request_stop(run)
        assert stop_requested(run)
    # A new explicit invocation acknowledges the previous stop request.
    with run_lock(run):
        assert not stop_requested(run)


def test_stop_checkpoints_at_optimizer_boundary(tmp_path, tiny_config, monkeypatch):
    from sdkb.operations import request_stop
    import torch
    run = tmp_path / 'run'
    original = torch.optim.AdamW.step
    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        request_stop(run)
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', step)
    result = train(tiny_config, run)
    assert result['steps'] == 1 and result['stopped_early']
    assert json.loads((resolve_checkpoint(run) / 'manifest.json').read_text())['step'] == 1


def test_archive_roundtrip_and_corruption(tmp_path, tiny_config):
    from sdkb.archiving import archive_checkpoint, restore_archive
    run, archive = tmp_path / 'run', tmp_path / 'archive'
    archive.mkdir()
    train(tiny_config, run)
    cp = resolve_checkpoint(run, verify=True)
    archived = archive_checkpoint(cp, archive, reserve_bytes=0)
    assert resolve_checkpoint(archive, verify=True).name == cp.name
    restored = tmp_path / 'restored'
    restore_archive(archive, restored)
    assert resolve_checkpoint(restored, verify=True).name == cp.name
    with pytest.raises(FileExistsError):
        restore_archive(archive, restored)
    with (archived / 'training_state.pt').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        restore_archive(archive, tmp_path / 'bad')


def test_incomplete_archive_is_never_published(tmp_path, tiny_config, monkeypatch):
    import sdkb.archiving as module
    run, archive = tmp_path / 'run', tmp_path / 'archive'
    archive.mkdir()
    train(tiny_config, run)
    def fail(*args, **kwargs):
        raise OSError('external disk disconnected')
    monkeypatch.setattr(module, 'copy_file', fail)
    with pytest.raises(OSError, match='disconnected'):
        module.archive_checkpoint(resolve_checkpoint(run), archive, reserve_bytes=0)
    assert not (archive / 'CURRENT').exists()
    assert resolve_checkpoint(run, verify=True)


def test_archive_backlog_is_bounded_and_active_copy_protected(tmp_path, monkeypatch):
    import sdkb.archiving as module
    entered, release = threading.Event(), threading.Event()
    def copy(source, *args, **kwargs):
        entered.set()
        release.wait(5)
        return source
    monkeypatch.setattr(module, 'archive_checkpoint', copy)
    archiver = module.CheckpointArchiver(tmp_path, reserve_bytes=0)
    first = tmp_path / 'step-1'
    archiver.submit(first)
    assert entered.wait(5)
    for i in range(2, 100):
        archiver.submit(tmp_path / f'step-{i}')
    assert archiver.protected_names() == {'step-1', 'step-99'}
    release.set()
    archiver.close(wait=True)
    assert not archiver.protected_names()


def test_low_disk_preserves_last_checkpoint(tmp_path, tiny_config, monkeypatch):
    import sdkb.archiving as module
    run = tmp_path / 'run'
    train(tiny_config, run, stop_after=1)
    current = (run / 'CURRENT').read_text()
    from collections import namedtuple
    Usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda p: Usage(100, 99, 1))
    with pytest.raises(OSError, match='free disk'):
        train(tiny_config, run, resume=True)
    assert (run / 'CURRENT').read_text() == current


def test_archive_retention_never_crosses_run_identity(tmp_path, tiny_config):
    from sdkb.archiving import archive_checkpoint
    archive = tmp_path / 'archive'
    archive.mkdir()
    first, second = tmp_path / 'first', tmp_path / 'second'
    train(tiny_config, first)
    train(tiny_config, second)
    archive_checkpoint(resolve_checkpoint(first), archive, reserve_bytes=0)
    current = (archive / 'CURRENT').read_text()
    with pytest.raises(ValueError, match='another'):
        archive_checkpoint(resolve_checkpoint(second), archive, keep=1, reserve_bytes=0)
    assert (archive / 'CURRENT').read_text() == current


def test_optional_tracking_keeps_identity_and_does_not_upload_payloads(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from sdkb.tracking import Tracking
    from sdkb.config import Config
    calls, logs, finished = [], [], []
    fake = SimpleNamespace(log=lambda row: logs.append(row), finish=lambda **kw: finished.append(kw),
                           define_metric=lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kw: calls.append(kw) or fake))
    config = Config()
    config.train.wandb_mode = 'offline'
    for _ in range(2):
        with Tracking(config, tmp_path) as tracker:
            tracker.log({'step': 1, 'loss': 2.})
    assert calls[0]['id'] == calls[1]['id']
    assert calls[0]['mode'] == 'offline' and calls[1]['resume'] == 'allow'
    assert all('source' not in row for row in logs) and len(finished) == 2
    assert Path(tmp_path / 'run-identity.json').exists()


def test_relocate_checkpoints_preserves_resume_and_frees_local_files(tmp_path, tiny_config):
    from sdkb.archiving import relocate_checkpoints
    run, destination = tmp_path / 'run', tmp_path / 'external'
    destination.mkdir()
    train(tiny_config, run, stop_after=1)
    old = resolve_checkpoint(run, verify=True).name
    result = relocate_checkpoints(run, destination)
    assert result['local_bytes_removed'] > 0
    assert (run / 'checkpoints').is_symlink()
    assert resolve_checkpoint(run, verify=True).name == old
    assert resolve_checkpoint(run).resolve().is_relative_to(destination)
    assert train(tiny_config, run, resume=True)['steps'] == tiny_config.train.steps
    assert not list(run.glob('.relocated-*'))


def test_relocate_refuses_corrupt_archive_and_running_owner(tmp_path, tiny_config):
    from sdkb.archiving import relocate_checkpoints
    from sdkb.operations import run_lock
    run, destination = tmp_path / 'run', tmp_path / 'external'
    destination.mkdir()
    train(tiny_config, run)
    with run_lock(run):
        with pytest.raises(RuntimeError, match='already running'):
            relocate_checkpoints(run, destination)
    import shutil
    checkpoint = resolve_checkpoint(run)
    copied = destination / checkpoint.name
    shutil.copytree(checkpoint, copied)
    (copied / 'model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        relocate_checkpoints(run, destination)
    assert not (run / 'checkpoints').is_symlink()
    resolve_checkpoint(run, verify=True)
