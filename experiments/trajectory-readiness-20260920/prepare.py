"""Prepare a bounded public-trajectory readiness sample entirely on external storage."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile


def prepare(root, cache):
    if not root.parent.is_dir() or not cache.is_dir():
        raise FileNotFoundError('Artifact parent and configured cache must already exist')
    if shutil.disk_usage(root.parent).free < 10*1024**3:
        raise OSError('Preparation requires a 10 GiB disk reserve')
    root.mkdir(exist_ok=True)
    for name, relative in [('HF_HOME', 'huggingface'), ('HF_HUB_CACHE', 'huggingface/hub'),
                           ('HF_DATASETS_CACHE', 'huggingface/datasets'), ('HF_XET_CACHE', 'huggingface/xet'),
                           ('XDG_CACHE_HOME', 'xdg'), ('TMPDIR', 'tmp/trajectory-readiness')]:
        path = cache/relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # tempfile may have cached its default before these explicit environment settings.
    tempfile.tempdir = os.environ['TMPDIR']
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Data preparation stopped; pinned input lock is retained')
    signal.signal(signal.SIGTERM, interrupted)
    from transformers import AutoTokenizer
    from sdkb.operations import atomic_json
    from sdkb.trajectories import pinned_spec, prepare_trajectories, file_sha256
    revision = '40cb2ad3b3044d5a41eee083a6103c8b523afa45'
    snapshot = cache/'huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots'/revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    budgets = dict(max_supports=4, max_targets_per_trajectory=3, recent_messages=1,
                   recent_tokens=512, task_tokens=160, max_source_tokens=512,
                   max_prompt_tokens=4096, max_target_tokens=512)
    declaration = dict(protocol='prefix', seed=191, validation_fraction=.2, budgets=budgets,
        tokenizer_revision=revision, tokenizer_files={p.name: file_sha256(p) for p in snapshot.iterdir()
            if p.name in {'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'}},
        request=dict(name='swe_smith', max_rows=512, shuffle_buffer=256, shuffle_seed=191),
        cache=str(cache), notice='Recorded public trajectories only; no live teacher calls or source-command execution. Row limit is not a network-byte limit. Preparation does not train a model or demonstrate agent success.')
    lock = root/'inputs.json'
    if lock.exists():
        old = json.loads(lock.read_text())
        if old['declaration'] != declaration:
            raise ValueError('Preparation inputs changed; use a new root')
        spec = old['pinned_source']
    else:
        spec = pinned_spec(declaration['request'])
        atomic_json(lock, dict(declaration=declaration, pinned_source=spec,
                               preparer_sha256=file_sha256(__file__)))
    if (root/'data').exists():
        manifest = json.loads((root/'data/manifest.json').read_text())
        if any(file_sha256(root/'data'/name) != digest for name, digest in manifest['sha256'].items()):
            raise ValueError('Prepared data changed')
        return
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=root) as tmp:
        pending = Path(tmp)/'data'
        prepare_trajectories([spec], tokenizer, budgets, pending, protocol='prefix',
                             seed=191, validation_fraction=.2)
        os.replace(pending, root/'data')
    print(json.dumps(json.loads((root/'data/manifest.json').read_text()), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.cache)
