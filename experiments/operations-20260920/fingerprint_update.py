"""One-update numerical comparison only: hash states instead of saving checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from sdkb.config import load_config
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256
import sdkb.training as training


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            visit((str(item.dtype), list(item.shape)))
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            visit(len(item))
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode() + b'\0')
    visit(value)
    return digest.hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config.train.steps = 1
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    def record(agent, optimizer, run, step, rng, cache, dataset_sha256, **_kwargs):
        state = {'step': step, 'dataset_sha256': dataset_sha256,
                 'model': {name: fingerprint(tensor) for name, tensor in agent.state_dict().items()},
                 'optimizer': fingerprint(optimizer.state_dict()),
                 'optimizer_names': fingerprint(optimizer._sdkb_parameter_names),
                 'sampler_rng': fingerprint(rng.getstate()),
                 'torch_rng': fingerprint(torch.get_rng_state()),
                 'cuda_rng': fingerprint(torch.cuda.get_rng_state_all()),
                 'gradients': {name: fingerprint(p.grad) for name, p in agent.named_parameters()}}
        records.append(state)
        atomic_json(run/f'fingerprint-{step}.json', state)
    training.save_checkpoint = record
    summary = training.train(config, args.output, init_from=args.source)
    if [r['step'] for r in records] != [0, 1]:
        raise ValueError('Unexpected numerical probe update sequence')
    atomic_json(args.output/'fingerprints.json', {
        'script_sha256': file_sha256(__file__), 'input_config_sha256': file_sha256(args.config),
        'source': str(args.source), 'environment': summary['environment'], 'records': records,
        'notice': 'One-update numerical probe; checkpoint saving is replaced only inside this process. '
                  'No model/optimizer checkpoint files are written, and this probe is not resumable training.',
    })
