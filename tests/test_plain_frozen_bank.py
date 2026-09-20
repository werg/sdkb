import inspect
import json
from pathlib import Path
import runpy

import pytest

from sdkb.agent import SDKBAgent
from sdkb.data import make_episode, save_episodes
from sdkb.operations import request_stop
from sdkb.store import DiskStore


def fixture(tmp_path, tiny_config, monkeypatch):
    namespace = runpy.run_path(str(Path(__file__).parents[1]/'scripts/build_frozen_bank.py'))
    build = namespace['run']
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'manifest.json').write_text('{}')
    data = tmp_path/'episodes.jsonl'
    save_episodes(data, [make_episode(0)])
    agent = SDKBAgent(tiny_config).eval()
    variables = inspect.unwrap(build).__globals__
    monkeypatch.setitem(variables, 'resolve_checkpoint', lambda *a, **kw: checkpoint)
    monkeypatch.setitem(variables, 'config_from_run', lambda *a: tiny_config)
    monkeypatch.setitem(variables, 'load_frozen_agent', lambda *a: (agent, None))
    return build, checkpoint, data, agent


def test_plain_bank_resume_reuses_committed_payloads(tmp_path, tiny_config, monkeypatch):
    build, checkpoint, data, agent = fixture(tmp_path, tiny_config, monkeypatch)
    output = tmp_path/'bank'
    assert build(checkpoint, data, output)['writer_calls'] > 0
    manifest = json.loads((output/'bank-manifest.json').read_text())
    monkeypatch.setattr(agent, 'produce', lambda *_: pytest.fail('Resume called writer'))
    assert build(checkpoint, data, output)['writer_calls'] == 0
    assert json.loads((output/'bank-manifest.json').read_text()) == manifest


def test_plain_bank_stop_rolls_back_incomplete_records(tmp_path, tiny_config, monkeypatch):
    build, checkpoint, data, agent = fixture(tmp_path, tiny_config, monkeypatch)
    output = tmp_path/'bank'
    produce = agent.produce
    def stop_after_first(ids):
        result = produce(ids)
        request_stop(output)
        return result
    monkeypatch.setattr(agent, 'produce', stop_after_first)
    with pytest.raises(RuntimeError, match='stop requested'):
        build(checkpoint, data, output)
    assert not (output/'bank-manifest.json').exists()
    with DiskStore(output/'bank.sqlite').connect() as db:
        assert db.execute('SELECT COUNT(*) FROM records').fetchone()[0] == 0


def test_plain_bank_stop_precedes_model_allocation(tmp_path, tiny_config, monkeypatch):
    build, checkpoint, data, _ = fixture(tmp_path, tiny_config, monkeypatch)
    output = tmp_path/'bank'
    request_stop(output)
    monkeypatch.setitem(inspect.unwrap(build).__globals__, 'load_frozen_agent', lambda *a: pytest.fail('Allocated after stop'))
    with pytest.raises(RuntimeError, match='stop requested'):
        build(checkpoint, data, output)
