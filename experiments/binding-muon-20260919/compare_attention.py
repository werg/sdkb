"""Post-freeze paired MLP/attention confirmation; run from an isolated checkout."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path.cwd() / 'scripts'))
from evaluate_causal_stages import wait_for_completion
from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import make_multiuse_world, save_episodes
from sdkb.evaluation import evaluate_transfer_run
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.trajectories import file_sha256


runs = {
    'mlp': Path('/archive/runs/binding-muon-selected-20260919'),
    'attention': Path('/archive/runs/binding-muon-attention-20260919'),
}
for run in runs.values():
    wait_for_completion(run)
root = Path('/archive/runs/muon-reader-confirmation-20260919')
root.mkdir(exist_ok=True)
episodes = root / 'episodes.jsonl'
if not episodes.exists():
    save_episodes(episodes, [e for i in range(32) for e in
                  make_multiuse_world(i, split='muon-reader-confirmation-20260919', bindings=2)])
episode_hash = file_sha256(episodes)
atomic_json(root / 'inputs.json', {
    'runs': {name: str(run) for name, run in runs.items()},
    'episodes_sha256': episode_hash,
    'protocol': 'Common new worlds generated after both curricula complete; frozen stored-only reads.',
    'scope': 'Text interpretation and final latent systems, equal source access; not matched compute.',
})
with run_lock(root, clear_stop=False):
    for name, run in runs.items():
        for stage in ('text_bootstrap', 'recurrent_joint'):
            if stop_requested(run) or stop_requested(root):
                raise RuntimeError(f'Stop requested for {run}')
            source = run / stage
            checkpoint = resolve_checkpoint(source, verify=True)
            identity = {'episodes_sha256': episode_hash,
                        'checkpoint_manifest_sha256': file_sha256(checkpoint / 'manifest.json')}
            output = root / f'{name}-{stage}.json'
            if output.exists():
                if json.loads(output.read_text())['identity'] != identity:
                    raise ValueError('Saved evaluation identity changed')
                continue
            # Isolate evaluation outputs from the completed training directory
            # and other stage comparators; read the captured immutable checkpoint.
            view = root / f'{name}-{stage}'
            view.mkdir(exist_ok=True)
            link = view / 'checkpoints'
            if not link.exists():
                link.symlink_to(checkpoint.parent.resolve(), target_is_directory=True)
            (view / 'CURRENT').write_text(checkpoint.name + '\n')
            report = evaluate_transfer_run(view, episodes, drop_supports=True,
                                           binding_counterfactuals=True)
            atomic_json(output, {'identity': identity,
                                'report': {k: v for k, v in report.items() if k != 'rows'}})
            print(json.dumps({'completed': str(output)}), flush=True)
