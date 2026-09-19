from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

from sdkb.data import make_multiuse_world

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from probe_payload_identifiers import split_identifiers, Readout
finally:
    sys.path.pop(0)


def test_identifier_probe_splits_complete_worlds_without_source_overlap():
    episodes = [e for i in range(4) for e in make_multiuse_world(i, bindings=2)]
    groups = split_identifiers(episodes, 1)
    assert len(groups['heldout']) == 2 and len(groups['train']) == 6
    assert not {e.environment for e in groups['train']} & {e.environment for e in groups['heldout']}
    assert not {e.required_ids[0] for e in groups['train']} & {e.required_ids[0] for e in groups['heldout']}


@pytest.mark.parametrize('corruption', ['future', 'label', 'duplicate'])
def test_identifier_probe_rejects_invalid_examples(corruption):
    episodes = [e for i in range(2) for e in make_multiuse_world(i, bindings=2)]
    index = next(i for i, e in enumerate(episodes) if e.task_family == 'multiuse/identifier')
    if corruption == 'future':
        episodes[index] = replace(episodes[index], query_time=0)
    elif corruption == 'label':
        episodes[index] = replace(episodes[index], answer='api_zzzzzz')
    else:
        episodes.append(episodes[index])
    with pytest.raises(ValueError):
        split_identifiers(episodes, 1)


def test_readout_normalization_is_fixed_from_training_values():
    values = torch.arange(24.).reshape(4, 6)
    model = Readout(values, 8)
    before = {k: v.clone() for k, v in model.named_buffers()}
    model(torch.full((2, 6), 1e3)).sum().backward()
    assert all(torch.equal(dict(model.named_buffers())[k], v) for k, v in before.items())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
