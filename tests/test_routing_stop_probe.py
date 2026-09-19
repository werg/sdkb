"""STOP must follow the complete required group, with all gradient paths intact."""
import importlib.util
import itertools
from pathlib import Path

import torch


def test_stop_objective_matches_explicit_sequence_probabilities(monkeypatch):
    path = Path(__file__).parents[1] / 'scripts/probe_routing_stop.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('routing_stop_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = torch.tensor([[2., -1., .1, .2], [1., 0., 3., -2.]], dtype=torch.float64, requires_grad=True)
    stop = torch.tensor([.4, -.5], dtype=torch.float64, requires_grad=True)
    required, lengths = torch.tensor([[0, 0], [0, 2]]), torch.tensor([1, 2])
    actual = module.stop_pair_loss(scores, stop, required, lengths)
    full = torch.cat((scores, stop[:, None]), -1)
    reference = []
    for row, target, count in zip(full, required, lengths, strict=True):
        probabilities = []
        for order in itertools.permutations(target[:count].tolist()):
            available = torch.ones(5, dtype=torch.bool)
            total = row.sum() * 0
            for index in (*order, 4):
                total = total + row[index] - row.masked_fill(~available, -torch.inf).logsumexp(-1)
                available = available.clone()
                available[index] = False
            probabilities.append(total)
        reference.append(-torch.logsumexp(torch.stack(probabilities), 0))
    expected = torch.stack(reference).mean()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(actual, (scores, stop), retain_graph=True)
    b = torch.autograd.grad(expected, (scores, stop))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    assert module.select_records(torch.tensor([[2., 1., 0.], [2., 1., 0.], [2., 1., 0.]]),
                                 torch.tensor([3., 1.5, 1.]), 2) == [[], [0], [0, 1]]
