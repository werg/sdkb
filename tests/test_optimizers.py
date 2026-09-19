import copy

import pytest
import torch
from safetensors.torch import load_file

from sdkb.agent import SDKBAgent
from sdkb.checkpoints import resolve_checkpoint
from sdkb.optimizers import make_optimizer
from sdkb.training import train


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_state_equal(left, right)
    else:
        assert a == b


@pytest.mark.skipif(not hasattr(torch.optim, 'Muon'), reason='Native Muon unavailable in this Torch')
def test_muon_ownership_and_complete_resume(tmp_path, tiny_config, monkeypatch):
    from sdkb.operations import request_stop
    from sdkb.replay import ReplayTape
    config = copy.deepcopy(tiny_config)
    config.train.optimizer = 'muon'
    config.train.steps = 3
    config.train.gradient_accumulation = 2
    config.train.live_fraction = .5
    config.train.checkpoint_every = 1000
    config.memory.noise_std = .02
    agent = SDKBAgent(config)
    optimizer = make_optimizer(agent)
    owned = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(owned) == len(set(owned))
    assert set(owned) == {id(p) for p in agent.parameters() if p.requires_grad}
    muon_ids = {id(p) for g in optimizer.optimizers['muon'].param_groups for p in g['params']}
    assert muon_ids
    assert id(agent.backbone.base.embedding.weight) not in muon_ids
    assert id(agent.write_slots) not in muon_ids
    full, partial = tmp_path / 'full', tmp_path / 'partial'
    train(config, full)
    original = ReplayTape.backward
    def stop(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        request_stop(partial)
        return result
    monkeypatch.setattr(ReplayTape, 'backward', stop)
    assert train(config, partial)['saved_microbatches'] == 1
    monkeypatch.setattr(ReplayTape, 'backward', original)
    train(config, partial, resume=True)
    a, b = resolve_checkpoint(full), resolve_checkpoint(partial)
    assert_state_equal(load_file(str(a / 'model.safetensors')), load_file(str(b / 'model.safetensors')))
    a_state = torch.load(a / 'training_state.pt', weights_only=True)
    b_state = torch.load(b / 'training_state.pt', weights_only=True)
    assert_state_equal(a_state, b_state)


def test_resume_rejects_optimizer_and_learning_rate_changes(tmp_path, tiny_config):
    config = copy.deepcopy(tiny_config)
    run = tmp_path / 'run'
    train(config, run, stop_after=1)
    config.train.optimizer = 'muon'
    with pytest.raises(ValueError, match='hyperparameters'):
        train(config, run, resume=True)
    config.train.optimizer = 'adamw'
    config.train.learning_rate *= 2
    with pytest.raises(ValueError, match='hyperparameters'):
        train(config, run, resume=True)


def test_operational_policy_preserves_training_config(tmp_path, tiny_config):
    from sdkb.operations import configure_checkpoints
    run = tmp_path / 'run'
    train(tiny_config, run, stop_after=1)
    configure_checkpoints(run, checkpoint_every=1000, archive_dir=None)
    train(tiny_config, run, resume=True)
    from sdkb.training import config_from_run
    saved = config_from_run(run)
    assert saved.train.checkpoint_every == 1000
    assert saved.train.learning_rate == tiny_config.train.learning_rate
