"""Generate without candidates from the completed, identity-checked routing banks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate_stored_generation import evaluate
from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import atomic_json, run_lock, stop_requested
from sdkb.trajectories import file_sha256


def generate(study):
    inputs = json.loads((study / 'inputs.json').read_text())
    root = study / 'confirmation'
    episodes = root / 'episodes.jsonl'
    runs = {'source': Path(inputs['source_checkpoint']),
            **{k: Path(v['run']) for k, v in inputs['configs'].items()}}
    with run_lock(root / 'generation', clear_stop=False):
        for name, run in runs.items():
            if stop_requested(study) or stop_requested(root / 'generation'):
                raise RuntimeError('Generation stopped')
            checkpoint = resolve_checkpoint(run, verify=True)
            summary = json.loads((root / name / 'summary.json').read_text())
            if (summary['checkpoint_manifest_sha256'] != file_sha256(checkpoint / 'manifest.json')
                    or summary['episodes_sha256'] != file_sha256(episodes)
                    or summary['routing']['read_budget'] != 2):
                raise ValueError('Frozen evaluation identity or budget differs')
            output = root / f'{name}-generation.json'
            if output.exists():
                raise FileExistsError(output)
            result = evaluate(checkpoint, root / name / 'bank.sqlite', episodes,
                              8, 24, learned_world=True, read_budget=2)
            atomic_json(output, result)
            print(json.dumps({'completed': name, 'by_family': result['by_family']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    generate(args.study.resolve())
