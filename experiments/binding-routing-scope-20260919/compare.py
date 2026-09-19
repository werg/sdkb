"""Compare frozen joint versus projection-only routing on new shared worlds."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate_binding_context import evaluate
from evaluate_causal_stages import wait_for_completion
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import make_multiuse_world, save_episodes
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.trajectories import file_sha256


def compare(study: Path):
    wait_for_completion(study)
    inputs = json.loads((study / 'inputs.json').read_text())
    runs = {'source': Path(inputs['source_checkpoint']),
            **{k: Path(v['checkpoint']) for k, v in inputs['controls'].items()},
            **{k: Path(v['run']) for k, v in inputs['configs'].items()}}
    root = study / 'confirmation'
    with run_lock(root, clear_stop=False):
        if stop_requested(study) or stop_requested(root):
            raise RuntimeError('Comparison stopped')
        root.mkdir(exist_ok=True)
        episodes = root / 'episodes.jsonl'
        if not episodes.exists():
            save_episodes(episodes, [e for i in range(32) for e in make_multiuse_world(
                i, split='routing-scope-confirmation-20260919', bindings=2)])
        episode_hash = file_sha256(episodes)
        atomic_json(root / 'inputs.json', {'episodes_sha256': episode_hash,
                    'protocol': 'New common worlds after both arms complete; frozen stored-only reads.'})
        for name, run in runs.items():
            if stop_requested(study) or stop_requested(root):
                raise RuntimeError('Comparison stopped')
            checkpoint = resolve_checkpoint(run, verify=True)
            manifest_hash = file_sha256(checkpoint / 'manifest.json')
            if name == 'source' and manifest_hash != inputs['source_manifest_sha256']:
                raise ValueError('Source checkpoint changed')
            if name in inputs['controls'] and manifest_hash != inputs['controls'][name]['manifest_sha256']:
                raise ValueError('Control checkpoint changed')
            output = root / name
            if (output / 'summary.json').exists():
                saved = json.loads((output / 'summary.json').read_text())
                if (saved['episodes_sha256'] != episode_hash or
                        saved['checkpoint_manifest_sha256'] != manifest_hash):
                    raise ValueError('Evaluation identity changed')
            else:
                evaluate(checkpoint, episodes, output, learned_world=True, read_budget=2)
            print(json.dumps({'completed': name}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    compare(args.study.resolve())
