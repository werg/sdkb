import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from elm.checkpoints import resolve_checkpoint
from elm.store import DiskStore, StoredRecord
from elm.training import train


def test_interrupted_resume_matches_uninterrupted_with_cache_and_noise(tmp_path, tiny_config):
    config = copy.deepcopy(tiny_config)
    config.train.steps = 5
    config.train.train_worlds = 5
    config.train.live_fraction = 0.5
    config.train.checkpoint_every = 1
    config.memory.noise_std = 0.03
    config.memory.compaction = 'synthetic'
    config.memory.compact_records = 1
    config.memory.compaction_warmup = 0
    config.memory.compaction_probability = 0.5
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    train(config, full)
    partial = train(config, resumed, stop_after=2)
    assert partial['stopped_early'] and partial['steps'] == 2
    # These writes and log rows simulate work performed after the last commit point.
    DiskStore(resumed / 'training_cache.sqlite').put(StoredRecord(
        'uncommitted', torch.ones(12), torch.ones(24), namespace='train'))
    with (resumed / 'metrics.jsonl').open('a') as handle:
        handle.write('{"step": 999, "loss": 0}\n{"step":')
    result = train(config, resumed, resume=True)
    assert result['steps'] == 5
    a = load_file(str(resolve_checkpoint(full) / 'model.safetensors'))
    b = load_file(str(resolve_checkpoint(resumed) / 'model.safetensors'))
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    def rows(path):
        return [{k: v for k, v in json.loads(s).items() if k != 'elapsed_seconds'}
                for s in (path / 'metrics.jsonl').read_text().splitlines()]
    assert rows(full) == rows(resumed)
    with DiskStore(resumed / 'training_cache.sqlite').connect() as db:
        assert db.execute("SELECT count(*) FROM records WHERE record_id='uncommitted'").fetchone()[0] == 0
    assert len(list((resumed / 'checkpoints').glob('step-*'))) == 2


def test_partial_publish_keeps_old_checkpoint(tmp_path, tiny_config, monkeypatch):
    import elm.checkpoints as checkpoints
    run = tmp_path / 'run'
    original = checkpoints._atomic_text
    writes = 0
    def fail(path, text):
        nonlocal writes
        if Path(path).name == 'CURRENT':
            writes += 1
            if writes == 2:
                raise OSError('simulated interrupted pointer update')
        return original(path, text)
    monkeypatch.setattr(checkpoints, '_atomic_text', fail)
    with pytest.raises(OSError, match='simulated'):
        train(tiny_config, run)
    cp = resolve_checkpoint(run, verify=True)
    assert json.loads((cp / 'manifest.json').read_text())['step'] == 0
    monkeypatch.setattr(checkpoints, '_atomic_text', original)
    assert train(tiny_config, run, resume=True)['steps'] == tiny_config.train.steps


def test_detects_corrupt_optimizer_before_loading(tmp_path, tiny_config):
    run = tmp_path / 'run'
    train(tiny_config, run)
    path = resolve_checkpoint(run) / 'training_state.pt'
    with path.open('ab') as handle:
        handle.write(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        resolve_checkpoint(run, verify=True)


def test_raw_to_paired_compaction_warm_start(tmp_path, tiny_config):
    raw = tmp_path / 'raw'
    train(tiny_config, raw)
    compact = copy.deepcopy(tiny_config)
    compact.memory.compaction = 'synthetic'
    compact.memory.compact_records = 1
    compact.memory.compaction_objective = 'paired'
    compact.memory.compaction_warmup = 0
    compact.memory.compaction_probability = 1.
    compact.train.steps = 1
    run = tmp_path / 'compact'
    result = train(compact, run, init_from=raw)
    assert result['steps'] == 1
    assert result['last']['compact_nll'] > 0
    provenance = json.loads((run / 'initialization.json').read_text())
    assert provenance['optimizer_reset']
    assert all(name.startswith('compactor.') for name in provenance['missing_initialized'])
    with pytest.raises(ValueError, match='resume or warm-start'):
        train(compact, run, resume=True, init_from=raw)


def test_compactor_only_does_not_move_raw_system(tmp_path, tiny_config):
    raw = tmp_path / 'raw'
    train(tiny_config, raw)
    original = load_file(str(resolve_checkpoint(raw) / 'model.safetensors'))
    compact = copy.deepcopy(tiny_config)
    compact.memory.compaction = 'synthetic'
    compact.memory.compact_records = 1
    compact.memory.compaction_objective = 'paired'
    compact.memory.compaction_warmup = 0
    compact.memory.compaction_probability = 1.
    compact.train.optimization_scope = 'compactor'
    compact.train.live_fraction = 0.
    run = tmp_path / 'compact'
    train(compact, run, init_from=raw)
    after = load_file(str(resolve_checkpoint(run) / 'model.safetensors'))
    for name, tensor in original.items():
        torch.testing.assert_close(after[name], tensor, rtol=0, atol=0)
    initial = load_file(str(sorted((run / 'checkpoints').glob('step-000000000-*'))[0] / 'model.safetensors'))
    assert any(not torch.equal(initial[k], after[k]) for k in after if k.startswith('compactor.'))
