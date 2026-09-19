"""Prepare projection-only routing with the prior joint arm as a fixed control."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import yaml

from sdkb.checkpoints import resolve_checkpoint
from sdkb.config import load_config
from sdkb.trajectories import file_sha256


def prepare(previous, output):
    old = json.loads((previous / 'inputs.json').read_text())
    record = old['configs']['group_routing']
    if file_sha256(record['config']) != record['sha256']:
        raise ValueError('Prior config identity changed')
    config = load_config(record['config'])
    source = resolve_checkpoint(Path(old['source_checkpoint']), verify=True)
    control = resolve_checkpoint(Path(record['run']), verify=True)
    if (file_sha256(source / 'manifest.json') != old['source_manifest_sha256']
            or file_sha256(config.train.episodes_file) != old['episodes_sha256']
            or json.loads((control / 'manifest.json').read_text())['step'] != 800
            or config.train.steps != 800 or config.train.optimization_scope != 'all'
            or config.train.optimizer != 'muon' or config.train.routing_weight != 1):
        raise ValueError('Expected the completed matched Muon joint-routing arm')
    config.train.optimization_scope = 'routing'
    config.train.wandb_group = 'binding-muon-routing-scope'
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'routing_only.yaml'
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    result = {'source_checkpoint': str(source),
              'source_manifest_sha256': file_sha256(source / 'manifest.json'),
              'episodes_sha256': old['episodes_sha256'], 'optimizer_reset': True,
              'configs': {'routing_only': {'config': str(path), 'sha256': file_sha256(path),
                                           'run': str(output / 'routing_only')}},
              'order': ['routing_only'],
              'controls': {'joint_routing': {'checkpoint': str(control),
                           'manifest_sha256': file_sha256(control / 'manifest.json')}},
              'prior_study': str(previous), 'status': 'prepared; not launched'}
    (output / 'inputs.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous.resolve(), args.output.resolve())
