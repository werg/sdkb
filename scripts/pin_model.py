#!/usr/bin/env python3
"""Resolve a real HF commit into a new YAML file; never silently edits the input."""
import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
source, target = Path(args.config), Path(args.output)
if target.exists():
    raise FileExistsError(target)
raw = yaml.safe_load(source.read_text())
model = raw['model']
if model['backend'] != 'hf':
    raise ValueError('Only HF models have remote revisions')
info = HfApi().model_info(model['model_id'], revision=model.get('revision', 'main'))
if not info.sha:
    raise RuntimeError('No resolved commit returned')
model['revision'] = info.sha
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(raw, sort_keys=False))
print(f"Pinned {model['model_id']} to {info.sha}: {target}")
