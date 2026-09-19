"""Post-training common-world continuation comparison, with stored-only generation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate_causal_stages import wait_for_completion
from evaluate_stored_generation import evaluate as generate
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import make_multiuse_world, save_episodes
from sdkb.evaluation import evaluate_transfer_run
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.trajectories import file_sha256


def compare(study: Path):
    wait_for_completion(study)
    inputs = json.loads((study / 'inputs.json').read_text())
    runs = {'source': Path(inputs['source_checkpoint']),
            **{k: Path(v['run']) for k, v in inputs['configs'].items()}}
    root = study / 'confirmation'
    with run_lock(root, clear_stop=False):
        if stop_requested(study) or stop_requested(root):
            raise RuntimeError('Comparison stopped')
        root.mkdir(exist_ok=True)
        episodes = root / 'episodes.jsonl'
        if not episodes.exists():
            save_episodes(episodes, [e for i in range(32) for e in make_multiuse_world(
                i, split='muon-binding-continuation-confirmation-20260919', bindings=2)])
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
            identity = {'episodes_sha256': episode_hash,
                        'checkpoint_manifest_sha256': manifest_hash}
            view = root / name
            view.mkdir(exist_ok=True)
            link = view / 'checkpoints'
            if not link.exists():
                link.symlink_to(checkpoint.parent.resolve(), target_is_directory=True)
            (view / 'CURRENT').write_text(checkpoint.name + '\n')
            output = root / f'{name}.json'
            if output.exists():
                record = json.loads(output.read_text())
                if record['identity'] != identity:
                    raise ValueError('Evaluation identity changed')
            else:
                report = evaluate_transfer_run(view, episodes, drop_supports=True,
                                               binding_counterfactuals=True)
                record = {'identity': identity,
                          'report': {k: v for k, v in report.items() if k != 'rows'}}
                atomic_json(output, record)
            if stop_requested(study) or stop_requested(root):
                raise RuntimeError('Comparison stopped')
            generation = root / f'{name}-generation.json'
            if not generation.exists():
                bank = Path(record['report']['directory']) / 'bank.sqlite'
                atomic_json(generation, generate(view, bank, episodes, 8, 24))
            else:
                saved = json.loads(generation.read_text())['inputs']
                if any(saved[k] != v for k, v in identity.items()):
                    raise ValueError('Generation identity changed')
            print(json.dumps({'completed': name}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    compare(args.study.resolve())
