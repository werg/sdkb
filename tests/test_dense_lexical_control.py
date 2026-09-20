"""Dense lexical controls must rank persisted keys without source/target leakage."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def dense():
    path = Path(__file__).parents[1]/'scripts/evaluate_dense_lexical_routing.py'
    spec = importlib.util.spec_from_file_location('dense_lexical_control', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def records():
    return [dict(record_id=name, source_id=name, terms={word: 1}, namespace='global',
                 space='s0', generation='frozen-v1', domain='research', created_at=1)
            for name, word in [('opaque-a', 'alpha'), ('opaque-b', 'beta')]]


def test_dense_keys_cross_serialized_boundary_without_terms(dense, tmp_path):
    rows = records()
    dense.build(rows, tmp_path/'index', width=64, seed=11)
    metadata, keys, settings = dense.reopen(tmp_path/'index')
    assert settings == {'width': 64, 'seed': 11}
    assert keys.shape == (2, 64) and keys.dtype == np.float32
    assert keys.nbytes == 2*64*4
    assert all('terms' not in row for row in metadata)
    expected = dense.rank_dense(metadata, keys, 'alpha', **settings, query_time=2, top_k=1)
    assert expected[0]['record_id'] == 'opaque-a'
    rows[0]['terms'] = {'changed': 100}
    assert dense.rank_dense(metadata, keys, 'alpha', **settings, query_time=2, top_k=1) == expected
    with (tmp_path/'index'/'keys.npy').open('ab') as file:
        file.write(b'corruption')
    with pytest.raises(ValueError, match='digest'):
        dense.reopen(tmp_path/'index')


def test_dense_rank_enforces_visibility_before_search(dense):
    rows = records()
    keys = np.stack([dense.encode(row['terms'], width=64, seed=11) for row in rows])
    baseline = dense.rank_dense(rows, keys, 'alpha', width=64, seed=11, query_time=2, top_k=2)
    for field, value in [('created_at', 2), ('created_at', 99), ('domain', 'private'),
                         ('namespace', 'other'), ('space', 'other'), ('generation', 'old')]:
        hidden = dict(rows[0], record_id='hidden', **{field: value})
        assert dense.rank_dense(rows+[hidden], np.concatenate([keys, keys[:1]]), 'alpha',
                                width=64, seed=11, query_time=2, top_k=2) == baseline
    assert dense.rank_dense(rows, keys, '', width=64, seed=11, query_time=2, top_k=2) == []


def test_hash_projection_is_deterministic_and_opaque_ids_are_metadata(dense):
    a = dense.encode({'alpha': 2, 'beta': 1}, width=64, seed=11)
    assert np.array_equal(a, dense.encode({'beta': 1, 'alpha': 2}, width=64, seed=11))
    assert not np.array_equal(a, dense.encode({'alpha': 2, 'beta': 1}, width=64, seed=23))
    assert np.linalg.norm(a) == pytest.approx(1)
    rows = records()
    keys = np.stack([dense.encode(row['terms'], width=64, seed=11) for row in rows])
    old = dense.rank_dense(rows, keys, 'alpha', width=64, seed=11, query_time=2, top_k=1)
    renamed = [dict(row, record_id='renamed-'+row['record_id'], answer='inert annotation') for row in rows]
    new = dense.rank_dense(renamed, keys, 'alpha', width=64, seed=11, query_time=2, top_k=1)
    assert new == [dict(old[0], record_id='renamed-'+old[0]['record_id'])]
    for width in (0, True):
        with pytest.raises(ValueError):
            dense.encode({'alpha': 1}, width=width, seed=11)
