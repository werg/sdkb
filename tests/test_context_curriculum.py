"""Training-only source-utility selection preserves complete versioned episodes."""
import json
from pathlib import Path
import runpy

import pytest


def select():
    return runpy.run_path(str(Path(__file__).parents[1] /
                              'experiments/trajectory-context-curriculum-20260920/run.py'))['_select_lines']


def test_top_half_selection_preserves_raw_episode_bytes_and_order():
    lines = [json.dumps({'episode_id': str(i), 'source': {'version': i}, 'answer': 'target'}) + '\n'
             for i in range(4)]
    chosen, ids = select()(lines, {'0': .8, '1': -.1, '2': .9, '3': .2})
    assert ids == {'0', '2'}
    assert chosen == lines[::2]


def test_selection_rejects_missing_or_duplicate_scores():
    lines = [json.dumps({'episode_id': str(i)}) + '\n' for i in range(4)]
    with pytest.raises(ValueError, match='score identities'):
        select()(lines, {'0': 1., '1': 2.})
    with pytest.raises(ValueError, match='episode identities'):
        select()([lines[0], lines[0], lines[2], lines[3]],
                 {'0': 1., '1': 2., '2': 3., '3': 4.})
