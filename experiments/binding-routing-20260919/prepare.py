"""Seal the matched MLP routing-supervision continuation configs."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from sdkb.checkpoints import resolve_checkpoint
from sdkb.training import config_from_run
from sdkb.trajectories import file_sha256


def prepare(source: Path, output: Path):
    checkpoint = resolve_checkpoint(source, verify=True)
    config = config_from_run(checkpoint)
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    if (manifest['step'] != 1600 or manifest['step'] != config.train.steps or config.train.optimizer != 'muon'
            or config.train.arm != 'memory' or config.train.evidence_scope != 'required'
            or config.memory.reader != 'mlp' or config.model.freeze_backbone
            or config.model.backbone_train_scope != 'recurrent_core'):
        raise ValueError('Expected completed 1600-update selected-support Muon MLP core continuation')
    config.train.steps = 800
    config.train.seed = 37
    config.train.checkpoint_every = 1000
    config.train.keep_checkpoints = 2
    config.train.archive_dir = None
    config.train.wandb_mode = 'offline'
    config.train.wandb_group = 'binding-muon-routing'
    config.train.retrieval = 'learned'
    config.train.routing_warmup = 100
    config.memory.neighbors = [2]
    from sdkb.data import load_episodes
    from sdkb.routing import validate_routing_dataset
    validate_routing_dataset(config, load_episodes(config.train.episodes_file))
    output.mkdir(parents=True, exist_ok=False)
    configs = {}
    for name, weight in [('no_routing_loss', 0.0), ('group_routing', 1.0)]:
        arm = deepcopy(config)
        arm.train.routing_weight = weight
        arm.validate()
        path = output / f'{name}.yaml'
        path.write_text(yaml.safe_dump(asdict(arm), sort_keys=False))
        configs[name] = {'config': str(path), 'sha256': file_sha256(path),
                         'run': str(output / name)}
    record = {'source_checkpoint': str(checkpoint),
              'source_manifest_sha256': file_sha256(checkpoint / 'manifest.json'),
              'episodes_sha256': file_sha256(config.train.episodes_file),
              'optimizer_reset': True, 'configs': configs, 'order': list(configs),
              'status': 'prepared; not launched'}
    (output / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
