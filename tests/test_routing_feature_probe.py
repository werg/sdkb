"""The diagnostic's batched objective must match the actual routing objective."""
import importlib.util
from pathlib import Path

import torch
from sdkb.routing import group_plan_loss


def test_feature_probe_pair_objective_and_gradient():
    path = Path(__file__).parents[1] / 'scripts/probe_routing_features.py'
    spec = importlib.util.spec_from_file_location('feature_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = torch.tensor([[3., -2., 1., .5], [-1., 4., 0., 2.], [90., -90., 0., 1.]],
                          dtype=torch.float64, requires_grad=True)
    required = torch.tensor([[0, 2], [3, 3], [1, 2]])
    lengths = torch.tensor([2, 1, 2])
    actual = module.pair_loss(scores, required, lengths)
    expected = torch.stack([group_plan_loss(row, [tuple(indices[:count].tolist())])
                            for row, indices, count in zip(scores, required, lengths, strict=True)]).mean()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    a, = torch.autograd.grad(actual, scores, retain_graph=True)
    b, = torch.autograd.grad(expected, scores)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
