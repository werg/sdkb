"""Native stored-only intermediate capture and unchanged output checks."""
from dataclasses import replace
import json
from pathlib import Path
import runpy

import torch

from sdkb.data import load_episodes
from sdkb.evaluation_adapter import load_frozen_agent
from sdkb.operations import atomic_json
from sdkb.runtime import configure_memory
from sdkb.store import DiskStore
from sdkb.training import config_from_run

code = Path(__file__).parents[2]
feature = runpy.run_path(str(code/'scripts/probe_payload_identifiers.py'))['reader_feature']
old = runpy.run_path('/workspace/sdkb/.sdkb/freshness-readout/scripts/probe_payload_identifiers.py')['reader_feature']
root = Path('/archive/probes/freshness-readout-20260920')
inputs = json.loads((root/'inputs.json').read_text())
checkpoint = Path(inputs['models']['worlds-8192']['checkpoint'])
config = config_from_run(checkpoint)
configure_memory(config.train)
torch.set_num_threads(config.train.threads)
agent, _ = load_frozen_agent(config, checkpoint)
agent.requires_grad_(False)

def forbidden(*_args, **_kwargs):
    raise AssertionError('Source encoding or target tokenization invoked')

agent.produce = agent.target_ids = forbidden
if agent.compactor is not None:
    agent.compactor.forward = forbidden
store = DiskStore(root/'worlds-8192/bank/bank.sqlite')
episodes = [e for e in load_episodes(root/'episodes.jsonl') if e.task_family == 'multiuse/identifier']
episode = episodes[0]
other = next(e for e in episodes if e.environment != episode.environment)
assert episode.query == other.query
assert torch.equal(old(agent, store, episode), feature(agent, store, episode))
report = {'default_output_bitwise_equal_to_f9625f7': True, 'boundaries': {}}
for representation in ('reader_inputs', 'reader_state', 'reader'):
    metadata = {}
    value = feature(agent, store, episode, representation=representation, metadata=metadata)
    assert torch.equal(value, feature(agent, store, replace(episode, answer='api_ffffff'), representation=representation))
    zero = feature(agent, store, episode, representation=representation, zero_values=True)
    assert torch.equal(zero, feature(agent, store, other, representation=representation, zero_values=True))
    assert torch.isfinite(value).all()
    report['boundaries'][representation] = metadata | {'dimension': value.numel(),
        'target_independent': True, 'zero_values_world_independent': True}
output = Path('/archive/probes/reader-stages-20260920/preflight.json')
output.parent.mkdir(parents=True, exist_ok=True)
atomic_json(output, report)
print(json.dumps(report, indent=2))
