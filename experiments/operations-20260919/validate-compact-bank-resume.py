import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import evaluate_compact_transfer as evaluation
from sdkb.agent import SDKBAgent
from sdkb.trajectories import file_sha256
from sdkb.operations import atomic_json

root = Path('/archive/profiles/compact-resume-20260919')
output = root / 'evaluation'
kwargs = dict(source=Path('/archive/runs/binding-muon-continuation-20260919/recurrent_core'),
              fit=Path('/archive/probes/posthoc-compaction-20260919/mlp'), output=output,
              episodes_file=Path('/archive/probes/posthoc-compaction-fresh-20260919/episodes.jsonl'),
              worlds=1, reuse_decoder=True)
evaluation.stop_requested = lambda *_: True
try:
    evaluation.run(**kwargs)
except RuntimeError as exc:
    assert 'Stored confirmation stopped' in str(exc)
else:
    raise AssertionError('Expected controlled interruption after all banks commit')
assert not (output / 'raw.json').exists()
bank_hashes = {p.name: file_sha256(p) for p in output.glob('*.sqlite')}
def forbidden(*_args, **_kwargs):
    raise AssertionError('Resume called frozen writer')
SDKBAgent.produce = forbidden
evaluation.stop_requested = lambda *_: False
evaluation.run(**kwargs)
assert {p.name: file_sha256(p) for p in output.glob('*.sqlite')} == bank_hashes
reports = {p.name: file_sha256(p) for p in output.glob('*.json')}
evaluation.run(**kwargs)
assert {p.name: file_sha256(p) for p in output.glob('*.json')} == reports
atomic_json(root / 'result.json', {'code_commit': '897d657', 'model': 'native LFM BF16',
    'interrupted_after_bank_commit': True, 'resumed_with_writer_forbidden': True,
    'stored_bank_sha256_unchanged': bank_hashes, 'completed_restart_preserved_reports': reports,
    'driver_sha256': file_sha256(__file__)})
print(json.dumps({'native_resume': 'passed'}), flush=True)
