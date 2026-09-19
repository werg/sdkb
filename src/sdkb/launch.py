"""Synchronous staged SDKB training with immutable inputs and resumable stages."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil
import tempfile
import time

import yaml
from .backbones import ByteTokenizer
from .config import load_config
from .data import make_boolean_world, make_multiuse_world, save_episodes
from .episode_index import EpisodeIndex
from .text import render_prompt
from .trajectories import prepare_trajectories, pinned_spec, digest, file_sha256


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(pending, path)


def load_recipe(path):
    recipe = yaml.safe_load(Path(path).read_text())
    allowed = {'base_config', 'protocol', 'seed', 'validation_fraction', 'sources',
               'preparation', 'stages', 'evaluation', 'causal_train_worlds'}
    if not isinstance(recipe, dict) or set(recipe) - allowed:
        raise ValueError('Unknown recipe fields')
    if recipe.get('protocol') not in {'prefix', 'cross_experience', 'causal'}:
        raise ValueError('Invalid protocol')
    names = []
    if not recipe.get('stages'):
        raise ValueError('At least one stage required')
    for stage in recipe['stages']:
        if set(stage) - {'name', 'arm', 'steps', 'freeze_backbone', 'init_from', 'oracle_anchor_weight', 'compaction'}:
            raise ValueError('Unknown stage fields')
        if not stage['name'].replace('_', '').isalnum() or stage['name'] in names:
            raise ValueError('Stage names must be unique safe identifiers')
        if stage.get('init_from') and stage['init_from'] not in names:
            raise ValueError('Stage dependency must precede stage')
        if stage['steps'] < 1:
            raise ValueError('Positive stage steps required')
        names.append(stage['name'])
    return recipe


def _tokenizer(config):
    if config.model.backend == 'tiny':
        return ByteTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.model.model_id, revision=config.model.revision, trust_remote_code=False)


def prepare_launch(recipe_path, output, *, resume=False):
    recipe = load_recipe(recipe_path)
    output = Path(output).resolve()
    base = load_config(recipe['base_config'])
    identity = digest({'recipe': recipe, 'base_config': asdict(base)})
    manifest_path = output / 'launch.json'
    if manifest_path.exists():
        if not resume:
            raise FileExistsError('Launch exists; use --resume')
        old = json.loads(manifest_path.read_text())
        if old['input_identity'] != identity:
            raise ValueError('Recipe/base configuration changed; use a new output directory')
        for name, expected in old['prepared_checksums'].items():
            if file_sha256(output / name) != expected:
                raise ValueError(f'Prepared input changed: {name}')
        return old
    if output.exists() and not resume:
        raise FileExistsError('Output exists; use a fresh directory or resume interrupted preparation')
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'input-lock.json'
    if lock.exists() and json.loads(lock.read_text())['input_identity'] != identity:
        raise ValueError('Interrupted preparation recipe/base configuration changed')
    atomic_json(lock, {'input_identity': identity})
    lock = output / 'model-lock.json'
    if lock.exists():
        previous = json.loads(lock.read_text())
        if previous['model_id'] != base.model.model_id:
            raise ValueError('Model lock mismatch')
        base.model.revision = previous['revision']
    else:
        if base.model.backend == 'hf':
            from huggingface_hub import HfApi
            info = HfApi().model_info(base.model.model_id, revision=base.model.revision)
            if not info.sha:
                raise RuntimeError('Model revision could not be resolved')
            base.model.revision = info.sha
        atomic_json(lock, {'model_id': base.model.model_id, 'revision': base.model.revision, 'backend': base.model.backend})
    tokenizer = _tokenizer(base)
    base.train.seed = recipe.get('seed', 17)
    data = output / 'data'
    dataset_lock = output / 'datasets-lock.json'
    if recipe['protocol'] != 'causal':
        if dataset_lock.exists():
            locked_sources = json.loads(dataset_lock.read_text())
        else:
            locked_sources = [pinned_spec(request) for request in recipe['sources']]
            atomic_json(dataset_lock, locked_sources)
    if not (data / 'manifest.json').exists():
        with tempfile.TemporaryDirectory(prefix='.data-', dir=output) as temp:
            pending = Path(temp) / 'data'
            if recipe['protocol'] == 'causal':
                pending.mkdir()
                worlds = recipe.get('causal_train_worlds', 128)
                for split, count in [('train', worlds), ('validation', max(1, recipe.get('evaluation', {}).get('causal_worlds', 32)))]:
                    examples = [e for i in range(count) for e in make_boolean_world(
                        i, split=f'causal-{split}-{base.train.seed}', operations=('a', 'b', 'xor'))]
                    save_episodes(pending / f'{split}.jsonl', examples)
                atomic_json(pending / 'manifest.json', {'protocol': 'causal', 'worlds': worlds})
            else:
                budgets = recipe.get('preparation', {}) | {k: getattr(base.train, k) for k in
                    ('max_source_tokens', 'max_prompt_tokens', 'max_target_tokens')}
                prepare_trajectories(locked_sources, tokenizer, budgets, pending, protocol=recipe['protocol'],
                                     seed=base.train.seed, validation_fraction=recipe.get('validation_fraction', .1))
            if data.exists():
                raise ValueError('Incomplete data directory; inspect it before reusing output')
            shutil.move(str(pending), str(data))
    for split in ('train', 'validation'):
        for e in EpisodeIndex(data / f'{split}.jsonl'):
            if len(tokenizer.encode(e.answer, add_special_tokens=False)) + int(tokenizer.eos_token_id is not None) > base.train.max_target_tokens:
                raise ValueError('Prepared target overflow')
            support_text = '\n'.join(s.text for s in e.supports if s.record_id in e.required_ids)
            if len(tokenizer.encode(render_prompt(tokenizer, e.query, support_text), add_special_tokens=False)) > base.train.max_prompt_tokens:
                raise ValueError('Prepared oracle-text overflow')
    stages = []
    for stage in recipe['stages']:
        config = deepcopy(base)
        config.train.episodes_file = str(data / 'train.jsonl')
        config.train.arm, config.train.steps = stage['arm'], stage['steps']
        config.model.freeze_backbone = stage.get('freeze_backbone', False)
        config.train.oracle_anchor_weight = stage.get('oracle_anchor_weight', 0.)
        if stage.get('compaction'):
            config.memory.compaction = 'synthetic'
            config.memory.compaction_probability, config.memory.compaction_warmup = 1., 0
            config.train.optimization_scope = 'compactor'
        config.validate()
        name = stage['name']
        path = output / f'{name}.yaml'
        path.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        stages.append(dict(name=name, config=str(path), run=str(output / name),
                           init_from=str(output / stage['init_from']) if stage.get('init_from') else None))
    files = ['input-lock.json', 'model-lock.json', 'data/train.jsonl', 'data/validation.jsonl', 'data/manifest.json']
    if dataset_lock.exists():
        files.append('datasets-lock.json')
    if (data / 'normalized.jsonl').exists():
        files.append('data/normalized.jsonl')
    files += [s['name'] + '.yaml' for s in stages]
    manifest = dict(format=1, project='SDKB', input_identity=identity, recipe=recipe, stages=stages,
                    resolved_model_revision=base.model.revision, backend=base.model.backend,
                    prepared_checksums={f: file_sha256(output / f) for f in files}, prepared_at_unix=time.time())
    atomic_json(manifest_path, manifest)
    return manifest


def launch(recipe_path, output, *, resume=False, prepare_only=False):
    manifest = prepare_launch(recipe_path, output, resume=resume)
    output = Path(output).resolve()
    if prepare_only:
        return {'status': 'prepared', 'manifest': str(output / 'launch.json'), 'stages': manifest['stages']}
    from .probes import model_probe
    from .training import train
    from .checkpoints import resolve_checkpoint
    from .trajectory_eval import evaluate_teacher_run
    probe = output / 'model-probe.json'
    if not probe.exists():
        atomic_json(probe, model_probe(load_config(manifest['stages'][0]['config'])))
    for stage in manifest['stages']:
        run = Path(stage['run'])
        config, completed = load_config(stage['config']), 0
        if (run / 'CURRENT').exists():
            checkpoint = resolve_checkpoint(run, verify=True)
            completed = json.loads((checkpoint / 'manifest.json').read_text())['step']
            if completed > config.train.steps:
                raise ValueError('Stage advanced outside its immutable launch plan')
        if completed < config.train.steps:
            if run.exists() and not (run / 'CURRENT').exists():
                raise RuntimeError(f'Uncommitted training directory {run}; inspect before retrying')
            result = train(config, run, resume=(run / 'CURRENT').exists(),
                           init_from=None if (run / 'CURRENT').exists() else stage['init_from'])
            if result['stopped_early']:
                return dict(status='checkpointed', stage=stage['name'], steps=result['steps'])
        report = output / (stage['name'] + '-evaluation.json')
        if not report.exists():
            atomic_json(report, evaluate_teacher_run(run, output / 'data/validation.jsonl',
                        max_episodes=manifest['recipe'].get('evaluation', {}).get('max_episodes', 64)))
    last = manifest['stages'][-1]
    count = manifest['recipe'].get('evaluation', {}).get('causal_worlds', 32)
    split = 'post-freeze-' + manifest['input_identity'][:12]
    if count:
        from .evaluation import evaluate_transfer_run
        if not (output / 'causal-evaluation.json').exists():
            path = output / 'fresh-causal.jsonl'
            save_episodes(path, [e for i in range(count) for e in make_boolean_world(i, split=split, operations=('a', 'b', 'xor'))])
            result = evaluate_transfer_run(last['run'], path, boolean_counterfactuals=True, drop_supports=True)
            atomic_json(output / 'causal-evaluation.json', {k: v for k, v in result.items() if k != 'rows'})
        # Independent marker: a failed binding evaluation does not silently get skipped on resume.
        if not (output / 'multiuse-evaluation.json').exists():
            path = output / 'fresh-multiuse.jsonl'
            save_episodes(path, [e for i in range(count) for e in make_multiuse_world(i, split=split, bindings=2)])
            result = evaluate_transfer_run(last['run'], path)
            atomic_json(output / 'multiuse-evaluation.json', {k: v for k, v in result.items() if k != 'rows'})
    return dict(status='complete', output=str(output), stages=[s['name'] for s in manifest['stages']],
                scientific_claim='Inspect stored-memory and counterfactual effects; execution alone is not evidence of transfer.')
