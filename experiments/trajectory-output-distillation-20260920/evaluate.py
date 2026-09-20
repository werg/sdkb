"""Score completed matched arms from frozen stored banks at trained depth two."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdkb.checkpoints import resolve_checkpoint
from sdkb.depth_eval import evaluate_depths
from sdkb.trajectories import file_sha256


def evaluate(root: Path) -> dict:
    lock = json.loads((root / 'inputs.json').read_text())
    if file_sha256(Path(lock['validation'])) != lock['validation_sha256']:
        raise ValueError('Validation bytes changed')
    results = {}
    for arm in ('control', 'distilled'):
        checkpoint = resolve_checkpoint(root / arm, verify=True)
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        if manifest['step'] != lock['steps'] or manifest['dataset_sha256'] != lock['train_sha256']:
            raise ValueError(f'{arm} is not a completed matched arm')
        result = evaluate_depths(root / arm, lock['validation'], root / f'{arm}-depth',
                                 depths=(2,), protocol='teacher', max_episodes=119)
        results[arm] = {'status': result['status'], 'directory': str(root / f'{arm}-depth')}
        if result['status'] != 'complete':
            break
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.root), indent=2))
