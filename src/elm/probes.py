"""Backbone preflight: execute public embeddings, causality and writer/read gradients."""
from __future__ import annotations

import torch

from .agent import MemoryAgent
from .training import autocast_context, environment_report, resource_report


def model_probe(config) -> dict:
    config.validate()
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(config.train.seed)
    agent = MemoryAgent(config).to(config.train.device)
    agent.eval()
    ids = agent.text_ids('An earlier experience has a retry rule.', source=True)
    with torch.no_grad(), autocast_context(config):
        embeddings = agent.backbone.embed(ids)
        mask = torch.ones_like(ids)
        base = agent.backbone.base.core(embeddings, mask)
        once = agent.backbone.hidden(embeddings, mask, loops=1)
        twice = agent.backbone.hidden(embeddings, mask, loops=2)
        error_one = float((base - once).abs().max())
        error_two = float((once - twice).abs().max())
        cut = embeddings.shape[1] // 2
        changed = embeddings.clone()
        changed[:, cut:] += torch.randn_like(changed[:, cut:]) * .1
        modified = agent.backbone.hidden(changed, mask, loops=2)
        causal_error = float((twice[:, :cut] - modified[:, :cut]).abs().max())
    tolerance = .03 if config.train.precision == 'bf16' else 1e-4
    if max(error_one, error_two, causal_error) > tolerance:
        raise AssertionError(f'Backbone identity/causality failed: {error_one}, {error_two}, {causal_error}')
    agent.train()
    with autocast_context(config):
        records = [agent.produce(agent.text_ids(text, source=True)) for text in
                   ['Prior experience: restore state before retry.', 'Prior experience: retry is permitted.']]
        result = agent(agent.prompt_ids('What should happen next?'), agent.target_ids('Restore and retry.'),
                       records, [0, 1], arm='memory')
    result.loss.backward()
    gradients = {}
    for name, parameter in [('write_slots', agent.write_slots), ('value_head', agent.value_head[-1].weight),
                            ('reader_output', (agent.reader.local[0] if hasattr(agent.reader, 'local') else agent.reader).output[-1].weight), ('memory_gate', agent.memory_gate)]:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f'Missing/nonfinite soft-memory gradient: {name}')
        gradients[name] = float(parameter.grad.norm())
    return {'environment': environment_report(), 'backend': config.model.backend, 'model_id': config.model.model_id,
            'resolved_revision': agent.resolved_revision, 'width': agent.width,
            'parameters': sum(p.numel() for p in agent.parameters()),
            'one_loop_identity_max_error': error_one, 'zero_gate_two_loop_max_error': error_two,
            'causal_prefix_max_error': causal_error, 'nll': float(result.nll.detach()),
            'gradient_norms': gradients, 'resources': resource_report(),
            'notice': 'Numerical preflight only; no capability training or cache-performance claim.'}
