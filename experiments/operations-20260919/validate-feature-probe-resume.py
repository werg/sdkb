import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import probe_read_count as probe
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256
root = Path('/archive/profiles/probe-recovery-20260919')
source = Path('/archive/runs/binding-muon-continuation-20260919/recurrent_core')
features = Path('/archive/probes/routing-breadth-20260919')
probe.run(source, features, root / 'reference', steps=40, representation='reader_query')
calls = 0
def stop(*_args):
    global calls
    calls += 1
    return calls == 22
probe.stop_requested = stop
try:
    probe.run(source, features, root / 'resumed', steps=40, representation='reader_query')
except RuntimeError as exc:
    assert 'stopped' in str(exc)
else:
    raise AssertionError('Expected emergency checkpoint at update21')
saved = torch.load(root / 'resumed/resume.pt', weights_only=True)
assert saved['step'] == 21
with (root / 'resumed/metrics.jsonl').open('a') as f:
    f.write('{"step": 40, "invalid": true}\n{"step":')
probe.stop_requested = lambda *_: False
probe.run(source, features, root / 'resumed', steps=40, representation='reader_query')
a = torch.load(root / 'reference/resume.pt', weights_only=True)
b = torch.load(root / 'resumed/resume.pt', weights_only=True)
def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b
for key in ('head','optimizer','sampling_rng','torch_rng','cuda_rng','parameter_names','step'):
    equal(a[key],b[key])
rows=[json.loads(line) for line in (root/'resumed/metrics.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==[20,40] and all('invalid' not in r for r in rows)
atomic_json(root/'result.json', {'commit':'dbb3435', 'native_bf16_muon':True,'interrupted_step':21,
    'final_step':40,'equal_keys':['head','optimizer','sampling_rng','torch_rng','cuda_rng','parameter_names','step'],
    'reconciled_metric_steps':[20,40], 'driver_sha256':file_sha256(__file__),
    'notice':'Frozen-feature count probe; no backbone training or new capability claim.'})
print(json.dumps({'native_probe_resume':'passed'}),flush=True)
