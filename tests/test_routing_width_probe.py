from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from probe_routing_width import widen_weights
    from probe_routing_stop import StopProbe
finally:
    sys.path.pop(0)


def weights():
    generator = torch.Generator().manual_seed(12)
    return {name: torch.randn(*shape, generator=generator) for name, shape in {
        'key.weight': (3, 6), 'query_head.weight': (3, 6),
        'address.weight': (3, 3), 'query_map.weight': (3, 3)}.items()}


def test_same_width_preserves_values_without_aliasing():
    original = weights()
    result = widen_weights(original, 3, torch.Generator().manual_seed(71))
    for name in original:
        assert torch.equal(result[name], original[name])
        assert result[name].data_ptr() != original[name].data_ptr()


def test_wider_initialization_preserves_blocks_and_can_learn_new_dimensions():
    original = weights()
    result = widen_weights(original, 7, torch.Generator().manual_seed(71))
    for name, value in original.items():
        assert torch.equal(result[name][:value.shape[0], :value.shape[1]], value)
    assert torch.count_nonzero(result['key.weight'][3:]) == 24
    assert torch.count_nonzero(result['query_head.weight'][3:]) == 0
    for name in ('address.weight', 'query_map.weight'):
        assert torch.equal(result[name][3:, 3:], torch.eye(4))
        assert torch.count_nonzero(result[name][:3, 3:]) == 0
        assert torch.count_nonzero(result[name][3:, :3]) == 0
    model = StopProbe(result, False)
    key = model.address(model.key(torch.ones(1, 6)))
    query = model.query_map(model.query_head(torch.arange(6.).unsqueeze(0)))
    torch.nn.functional.cosine_similarity(key, query).sum().backward()
    assert torch.count_nonzero(model.query_head.weight.grad[3:]) > 0


@pytest.mark.parametrize('case', ['narrow', 'missing', 'shape'])
def test_incompatible_width_initialization_fails(case):
    original = weights()
    width = 2 if case == 'narrow' else 7
    if case == 'missing':
        original.pop('address.weight')
    if case == 'shape':
        original['query_head.weight'] = torch.zeros(4, 6)
    with pytest.raises(ValueError):
        widen_weights(original, width, torch.Generator())
