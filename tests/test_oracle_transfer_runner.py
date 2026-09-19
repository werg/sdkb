import json
from pathlib import Path
import sys

import pytest

from sdkb.data import make_multiuse_world, save_episodes
from sdkb.training import train
from sdkb.trajectories import file_sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
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
    assert len(report['generation_rows']) == len(episodes)*5
    assert set(report['writes']) == {'global', 'permission', 'restoration'}
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
