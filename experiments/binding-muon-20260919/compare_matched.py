from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path.cwd() / 'scripts'))
from evaluate_causal_stages import wait_for_completion, compare
from sdkb.data import make_multiuse_world, save_episodes
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256

runs = [Path('/archive/runs/binding-muon-selected-20260919'), Path('/archive/runs/binding-muon-all-20260919')]
for run in runs:
    wait_for_completion(run)
root = Path('/archive/runs/muon-binding-confirmation-20260919')
root.mkdir(exist_ok=True)
episodes = root / 'episodes.jsonl'
if not episodes.exists():
    save_episodes(episodes, [e for i in range(32) for e in make_multiuse_world(i, split='muon-binding-confirmation-20260919', bindings=2)])
atomic_json(root / 'inputs.json', {'runs': [str(r) for r in runs], 'episodes_sha256': file_sha256(episodes),
    'protocol': 'Common held-out worlds created after both curricula complete; each stage writes its own frozen bank.'})
for run in runs:
    print(json.dumps(compare(run, episodes)), flush=True)
