"""Source-bound, address-only overlays for frozen routing-probe evaluation.

These are inference adapters, not standard resumable training checkpoints. The
probe's optimizer state remains in its own external artifact.
"""
from pathlib import Path

from safetensors.torch import load_model
import torch

from .agent import SDKBAgent
from .trajectories import file_sha256


def load_frozen_agent(config, checkpoint, *, routing_probe=None, independent_routing_query=False):
    checkpoint = Path(checkpoint)
    state = None
    if routing_probe is not None:
        state = torch.load(routing_probe, weights_only=True, map_location='cpu')
        identity = state['identity']
        if identity['checkpoint_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json'):
            raise ValueError('Routing probe source checkpoint differs')
        if state['step'] != identity['steps']:
            raise ValueError('Routing probe endpoint is incomplete')
        if config.memory.independent_routing_query or len(config.memory.payload_dims) != 1:
            raise ValueError('Probe overlay requires a shared-query, single-space source')
        if set(state['model']) != {'key.weight', 'address.weight', 'query_map.weight', 'query_head.weight'}:
            raise ValueError('Unexpected probe parameter ownership')
        config.memory.independent_routing_query = independent_routing_query
    elif independent_routing_query:
        raise ValueError('Supply a routing probe for this evaluation override')
    agent = SDKBAgent(config).to(config.train.device).eval()
    missing, unexpected = load_model(agent, str(checkpoint / 'model.safetensors'), strict=False,
                                     device=config.train.device)
    expected = {'routing_query_head.weight'} if state is not None and independent_routing_query else set()
    if set(missing) != expected or unexpected:
        raise ValueError('Frozen source model topology differs')
    if state is None:
        return agent, None
    weights = state['model']
    if not independent_routing_query and not torch.equal(agent.query_head.weight.detach().cpu(), weights['query_head.weight']):
        raise ValueError('Shared adapter would change the frozen reader query')
    for module, name in [(agent.key_head, 'key'), (agent.address_maps[0], 'address'),
                         (agent.query_maps[0], 'query_map')]:
        module.load_state_dict({'weight': weights[name + '.weight']})
    if independent_routing_query:
        agent.routing_query_head.load_state_dict({'weight': weights['query_head.weight']})
    return agent, {'probe_sha256': file_sha256(routing_probe), 'identity': identity,
                   'independent_routing_query': independent_routing_query,
                   'notice': 'Address-only overlay; base reader, payload writer and decoder remain frozen.'}
