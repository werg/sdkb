import importlib.util
from pathlib import Path

from safetensors.torch import load, save
import torch

from sdkb.compaction import SyntheticCompactor, contribution_loss
from sdkb.readers import SetReader


def test_compactor_training_uses_serialized_precision_and_frozen_reader():
    path = Path(__file__).parents[1] / 'scripts/probe_stored_compaction.py'
    spec = importlib.util.spec_from_file_location('stored_compaction_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compactor = SyntheticCompactor(24, 24, 1)
    reader = SetReader(24, 12, 32, width=24, slots=2, rounds=2).eval().requires_grad_(False)
    raw, query = torch.randn(3, 2, 24).bfloat16().float(), torch.randn(3, 12)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        values, weights = module.serialized_codes(compactor, raw)
        stored = load(save({'values': values.detach().bfloat16(), 'weights': weights.detach()}))
        live = reader(values, query, weights).tokens
        persisted = reader(stored['values'].float(), query, stored['weights']).tokens
        torch.testing.assert_close(live, persisted, rtol=0, atol=0)
        torch.testing.assert_close(weights.sum(1), torch.full((3,), 2.), rtol=0, atol=0)
        loss = contribution_loss(reader, raw, torch.ones(3, 2), values, weights, query)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in reader.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in compactor.parameters())
