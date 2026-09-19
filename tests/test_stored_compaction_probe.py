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


def test_cached_single_read_task_gradient_matches_causal_prefix_plan(tiny_config):
    from sdkb.agent import SDKBAgent
    path = Path(__file__).parents[1] / 'scripts/probe_stored_compaction.py'
    spec = importlib.util.spec_from_file_location('task_compaction_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = tiny_config
    c.model.tiny_layers = 4
    c.model.recurrence_mode = 'middle_block'
    c.model.recurrent_start, c.model.recurrent_end = 1, 3
    c.model.loops, c.model.writer_loops = 3, 1
    c.memory.read_timing, c.memory.read_steps = 'loop_boundary', 1
    c.validate()
    agent = SDKBAgent(c).eval().requires_grad_(False)
    compactor = SyntheticCompactor(24, 24, 1)
    prompt, target = agent.prompt_ids('Use the stored rules.'), agent.target_ids('RETRY')
    raw = torch.randn(1, 2, 24).bfloat16().float()
    captured, tokens = [], None
    def provider(completed, query):
        nonlocal tokens
        if completed == 1:
            captured.append(query.detach().clone())
            values, weights = module.serialized_codes(compactor, raw)
            tokens = agent.reader(values, query, weights).tokens
        return tokens
    planned = agent.plan_loop_memory(prompt, provider)
    reference = agent.conditioned_nll(prompt, target, planned)
    expected = torch.autograd.grad(reference, tuple(compactor.parameters()))
    values, weights = module.serialized_codes(compactor, raw)
    tokens = agent.reader(values, captured[0], weights).tokens
    cached = module.single_read_memory(agent, prompt, tokens)
    actual = agent.conditioned_nll(prompt, target, cached)
    gradients = torch.autograd.grad(actual, tuple(compactor.parameters()))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    assert any(g.abs().sum() > 0 for g in gradients)
    assert all(p.grad is None for p in agent.parameters())
