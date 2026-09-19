"""Address-only diagnostic overlays must preserve the frozen composition model."""
import copy
import json

import pytest
import torch
from safetensors.torch import load_model

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.training import train
from sdkb.trajectories import file_sha256


def test_probe_overlay_preserves_reader_and_checks_source(tmp_path, tiny_config):
    from sdkb.evaluation_adapter import load_frozen_agent
    tiny_config.train.steps = 1
    source = tmp_path / 'source'
    train(tiny_config, source)
    checkpoint = resolve_checkpoint(source, verify=True)
    original = SDKBAgent(tiny_config).eval()
    load_model(original, str(checkpoint / 'model.safetensors'))
    weights = {'key.weight': original.key_head.weight.detach().clone(),
               'address.weight': original.address_maps[0].weight.detach().clone(),
               'query_map.weight': original.query_maps[0].weight.detach().clone(),
               'query_head.weight': original.query_head.weight.detach().clone() + .5}
    state = {'step': 3, 'identity': {'steps': 3, 'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json')},
             'model': weights}
    probe = tmp_path / 'probe.pt'
    torch.save(state, probe)
    agent, identity = load_frozen_agent(copy.deepcopy(tiny_config), checkpoint, routing_probe=probe,
                                       independent_routing_query=True)
    assert identity['probe_sha256'] == file_sha256(probe)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, agent.state_dict()[name], atol=0, rtol=0)
    torch.testing.assert_close(agent.routing_query_head.weight, weights['query_head.weight'], atol=0, rtol=0)
    with pytest.raises(ValueError, match='reader query'):
        load_frozen_agent(copy.deepcopy(tiny_config), checkpoint, routing_probe=probe)
    state['identity']['checkpoint_manifest_sha256'] = 'wrong'
    torch.save(state, probe)
    with pytest.raises(ValueError, match='source'):
        load_frozen_agent(copy.deepcopy(tiny_config), checkpoint, routing_probe=probe, independent_routing_query=True)
    assert json.loads((checkpoint / 'config.json').read_text())['memory']['independent_routing_query'] is False


def test_probe_choice_and_generation_share_bank_identity(tmp_path, tiny_config):
    import importlib.util
    from pathlib import Path
    from sdkb.data import make_multiuse_world, save_episodes
    tiny_config.train.steps = 1
    tiny_config.train.max_prompt_tokens = 1500
    source = tmp_path / 'source'
    train(tiny_config, source)
    checkpoint = resolve_checkpoint(source, verify=True)
    agent = SDKBAgent(tiny_config).eval()
    load_model(agent, str(checkpoint / 'model.safetensors'))
    probe = tmp_path / 'probe.pt'
    torch.save({'step': 1, 'identity': {'steps': 1, 'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json')},
                'model': {'key.weight': agent.key_head.weight, 'address.weight': agent.address_maps[0].weight,
                          'query_map.weight': agent.query_maps[0].weight, 'query_head.weight': agent.query_head.weight + .2}}, probe)
    def script(name):
        spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'scripts' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    choice, generation = script('evaluate_binding_context'), script('evaluate_stored_generation')
    episodes = tmp_path / 'episodes.jsonl'
    save_episodes(episodes, make_multiuse_world(23, bindings=2))
    output = tmp_path / 'evaluation'
    choice.evaluate(source, episodes, output, learned_world=True, routing_probe=probe, independent_routing_query=True)
    result = generation.evaluate(source, output / 'bank.sqlite', episodes, 1, 1, learned_world=True,
                                 routing_probe=probe, independent_routing_query=True)
    report = json.loads((output / 'results.json').read_text())
    expected = {r['episode']: r['selected_ids'] for r in report['rows'] if r['condition'] == 'all'}
    for row in result['rows']:
        if row['condition'] == 'all':
            assert row['selected_ids'] == expected[row['episode']]
    with pytest.raises(ValueError, match='bank routing adapter'):
        generation.evaluate(source, output / 'bank.sqlite', episodes, 1, 1, learned_world=True)
    report['routing_probe']['probe_sha256'] = 'changed'
    (output / 'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='bank routing adapter'):
        generation.evaluate(source, output / 'bank.sqlite', episodes, 1, 1, learned_world=True,
                            routing_probe=probe, independent_routing_query=True)


def test_legacy_episode_evaluator_uses_independent_routing_query(tmp_path, tiny_config, monkeypatch):
    from sdkb.data import make_episode, save_episodes
    from sdkb.store import DiskStore
    from sdkb.training import evaluate_episode_file
    tiny_config.memory.independent_routing_query = True
    tiny_config.train.retrieval = 'learned'
    tiny_config.train.steps = 1
    source = tmp_path / 'source'
    train(tiny_config, source)
    path = tmp_path / 'episodes.jsonl'
    save_episodes(path, [make_episode(3, distractors=1)])
    pair, search = SDKBAgent.query_pair, DiskStore.search
    expected = []
    def capture(self, *args, **kwargs):
        reader_query, routing_query = pair(self, *args, **kwargs)
        expected[:] = [self.query_maps[0](routing_query)[0]]
        return reader_query, routing_query
    def checked_search(self, query, **kwargs):
        torch.testing.assert_close(query, expected[0])
        return search(self, query, **kwargs)
    monkeypatch.setattr(SDKBAgent, 'query_pair', capture)
    monkeypatch.setattr(SDKBAgent, 'query', lambda *args: pytest.fail('Legacy path bypassed routing query'))
    monkeypatch.setattr(DiskStore, 'search', checked_search)
    evaluate_episode_file(source, path)
