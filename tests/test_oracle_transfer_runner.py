import json
from pathlib import Path
import sys

import pytest

from sdkb.data import make_multiuse_world, save_episodes
from sdkb.training import train
from sdkb.trajectories import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import evaluate_oracle_transfer as evaluator
    from evaluate_oracle_transfer import run
finally:
    sys.path.pop(0)


def test_oracle_runner_offline_banks_free_generation_and_completed_restart(tmp_path, tiny_config, monkeypatch):
    path = tmp_path/'episodes.jsonl'
    episodes = make_multiuse_world(9, bindings=1)
    save_episodes(path, episodes)
    tiny_config.train.steps = 1
    tiny_config.train.episodes_file = str(path)
    tiny_config.train.evidence_scope = 'required'
    source = tmp_path/'source'
    train(tiny_config, source)
    output = tmp_path/'evaluation'
    run(source, path, output)
    report = json.loads((output/'results.json').read_text())
    assert len(report['generation_rows']) == len(episodes)*6
    assert set(report['writes']) == {'global', 'permission', 'restoration', 'identifier'}
    assert report['inputs']['routing'] == 'oracle required sources'
    hashes = {name: file_sha256(output/name) for name in ('bank.sqlite', 'results.json')}
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed restart loaded a model')
    monkeypatch.setattr('evaluate_oracle_transfer.load_frozen_agent', forbidden)
    run(source, path, output)
    assert hashes == {name: file_sha256(output/name) for name in hashes}
    with (output/'bank.sqlite').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='bank changed'):
        run(source, path, output)


def test_endpoint_counterfactual_changes_only_endpoint_information():
    original = make_multiuse_world(8, bindings=2)
    changed = [evaluator.identifier_counterfactual(e) for e in original]
    shared_sources = {}
    for a, b in zip(original, changed):
        assert (a.episode_id, a.query, a.query_time, a.required_ids) == (b.episode_id, b.query, b.query_time, b.required_ids)
        assert (a.restore, a.allowed_capability, a.capability) == (b.restore, b.allowed_capability, b.capability)
        assert (a.answer != b.answer) == (a.task_family == 'multiuse/identifier')
        if a.task_family == 'multiuse/identifier':
            assert b.answer in b.choices
            assert b.answer in next(s.text for s in b.supports if s.record_id in b.required_ids)
        for old, new in zip(a.supports, b.supports):
            assert (old.record_id, old.created_at, old.kind) == (new.record_id, new.created_at, new.kind)
            if old.kind == 'restoration':
                assert old == new
            else:
                assert old.text.split('Its endpoint')[0] == new.text.split('Its endpoint')[0]
                assert old.text != new.text
            assert shared_sources.setdefault(new.record_id, new) == new
