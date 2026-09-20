from dataclasses import asdict
import json
from pathlib import Path
import runpy

import pytest
import yaml

from sdkb.agent import SDKBAgent
from sdkb.data import make_multiuse_world, save_episodes
from sdkb.operations import request_stop


def test_text_readiness_resume_reuses_rows_and_never_writes_sources(tmp_path, tiny_config, monkeypatch):
    helper = runpy.run_path(str(Path(__file__).parents[1]/'experiments/trajectory-readiness-20260920/evaluate_text.py'))
    tiny_config.train.arm = 'oracle_text'
    tiny_config.train.evidence_scope = 'available'
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(asdict(tiny_config)))
    episodes = tmp_path/'episodes.jsonl'
    save_episodes(episodes, make_multiuse_world(3, bindings=1)[:2])
    def forbidden(*args, **kwargs):
        raise AssertionError('Text readiness must never invoke the writer')
    monkeypatch.setattr(SDKBAgent, 'produce', forbidden)
    full, partial = tmp_path/'full', tmp_path/'partial'
    helper['run'](config, episodes, full)
    original = SDKBAgent.conditioned_nll
    calls = []
    def stop(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(True)
        request_stop(partial)
        return result
    monkeypatch.setattr(SDKBAgent, 'conditioned_nll', stop)
    helper['run'](config, episodes, partial)
    assert len(calls) == 1 and not (partial/'results.json').exists()
    assert len(json.loads((partial/'progress.json').read_text())['rows']) == 1
    def count(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SDKBAgent, 'conditioned_nll', count)
    helper['run'](config, episodes, partial)
    assert len(calls) == 4
    a, b = [json.loads((p/'results.json').read_text()) for p in (full, partial)]
    assert a['rows'] == b['rows'] and a['summary'] == b['summary']
    monkeypatch.setattr(SDKBAgent, '__init__', forbidden)
    helper['run'](config, episodes, partial)
    episodes.write_text(episodes.read_text()+'\n')
    with pytest.raises(ValueError, match='inputs changed'):
        helper['run'](config, episodes, partial)
