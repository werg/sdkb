"""Frozen text-control viability for individual-character questions."""
import argparse
import json
from pathlib import Path
import runpy

import torch

from sdkb.archiving import ensure_free
from sdkb.checkpoints import _atomic_text, resolve_checkpoint, stop_on_signal
from sdkb.data import load_episodes
from sdkb.evaluation_adapter import load_frozen_agent
from sdkb.frozen_scoring import FrozenScorer
from sdkb.operations import run_lock, stop_requested
from sdkb.training import autocast_context, config_from_run
from sdkb.trajectories import file_sha256


@torch.no_grad()
def run(root, loops):
    output = root/f'text-preflight-r{loops}'
    output.mkdir(exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        if stop_requested(output):
            raise RuntimeError('Text preflight stop requested')
        declared = json.loads((root/'preflight-inputs.json').read_text())
        source = resolve_checkpoint(Path(declared['source']), verify=True)
        corpus = root/'preflight-characters.jsonl'
        helper = Path(__file__).parents[2]/'scripts/evaluate_exact_detail.py'
        identity = {'source_manifest_sha256': file_sha256(source/'manifest.json'),
                    'episodes_sha256': file_sha256(corpus), 'script_sha256': file_sha256(__file__),
                    'selected_text_helper_sha256': file_sha256(helper), 'max_new_tokens': 24, 'loops': loops}
        if (identity['source_manifest_sha256'] != declared['source_manifest_sha256'] or
                identity['episodes_sha256'] != declared['character_sha256']):
            raise ValueError('Preflight source/corpus differs from declaration')
        selected_text = runpy.run_path(str(helper))['selected_text']
        episodes = [e for e in load_episodes(corpus) if e.task_family in
                    {'multiuse/identifier', 'identifier_character'}]
        expected = [(e, c) for e in episodes for c in ('text_required', 'none')]
        config = config_from_run(source)
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, source)
        agent.requires_grad_(False)
        agent.backbone.loops = loops
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Text preflight invoked a latent writer')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        scorer, rows, progress = FrozenScorer(agent), [], output/'progress.json'
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved['inputs'] != identity:
                raise ValueError('Text preflight progress identity differs')
            rows = saved['rows']
        if len(rows) > len(expected):
            raise ValueError('Extra text-preflight progress')
        for row, (e, c) in zip(rows, expected):
            if (row['episode'], row['condition'], row['answer']) != (e.episode_id, c, e.answer):
                raise ValueError('Text-preflight progress prefix differs')
        def save(path, value):
            ensure_free(output, config.train.min_free_disk_bytes)
            _atomic_text(path, json.dumps(value, indent=2)+'\n')
        with autocast_context(config):
            for e, condition in expected[len(rows):]:
                if stop['signal'] is not None or stop_requested(output):
                    raise RuntimeError('Text preflight stopped; completed strings are resumable')
                text = selected_text(e) if condition == 'text_required' else ''
                prompt = agent.prompt_ids(e.query, text)
                prediction = scorer.generate(prompt, None, max_new_tokens=24)
                rows.append({'episode': e.episode_id, 'environment': e.environment,
                             'task_family': e.task_family, 'position': e.provenance['endpoint_character_position'],
                             'condition': condition, 'answer': e.answer, 'prediction': prediction,
                             'exact_match': prediction.strip() == e.answer.strip(), 'prompt_tokens': prompt.numel()})
                save(progress, {'inputs': identity, 'rows': rows})
                if len(rows) % 32 == 0:
                    print(json.dumps({'completed': len(rows), 'total': len(expected)}), flush=True)
        summary = {}
        for family in ('multiuse/identifier', 'identifier_character'):
            for position in ([None] if family == 'multiuse/identifier' else [None, 1, 2, 3, 4, 5, 6]):
                name = family if position is None else family+'/'+str(position)
                summary[name] = {c: {'n': len(group := [r for r in rows if r['task_family'] == family and
                    r['condition'] == c and (position is None or r['position'] == position)]),
                    'correct': sum(r['exact_match'] for r in group)} for c in ('text_required', 'none')}
        save(output/'results.json', {'inputs': identity, 'summary': summary, 'rows': rows,
             'notice': 'Selected-text question-viability preflight, not latent-memory capability. No candidate '
                       'answers or target prefixes are supplied. Source commands are inert context.'})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--loops', type=int, choices=(1, 3), default=3)
    args = parser.parse_args()
    run(args.root, args.loops)
