import json
from pathlib import Path

import pytest

from elm.cli import main
from elm.config import load_config
from elm.diagnostics import compact_probe, io_benchmark


def test_doctor_cli(capsys):
    main(['doctor'])
    assert 'torch' in json.loads(capsys.readouterr().out)


def test_make_data_cli(tmp_path, capsys):
    target = tmp_path / 'episodes.jsonl'
    main(['make-data', '--output', str(target), '--count', '3'])
    assert len(target.read_text().splitlines()) == 3
    assert json.loads(capsys.readouterr().out)['count'] == 3
    with pytest.raises(FileExistsError):
        main(['make-data', '--output', str(target)])


def test_all_shipped_configs():
    root = Path(__file__).resolve().parents[1]
    for path in (root / 'configs').glob('*.yaml'):
        load_config(path)


def test_io_probe(tmp_path):
    result = io_benchmark(tmp_path / 'io.sqlite', records=8, reads=2, payload_dim=12,
                          key_dim=4, neighbors=2)
    assert result['ideal_value_bytes_per_read'] == 48
    assert result['fetch']['p50_ms'] >= 0


@pytest.mark.parametrize('kind', ['mlp', 'attention'])
def test_compactor_probe(kind):
    result = compact_probe(steps=2, reader_kind=kind)
    assert result['last_training_loss'] >= 0
    assert set(result['after']) == {'redundant', 'independent'}
