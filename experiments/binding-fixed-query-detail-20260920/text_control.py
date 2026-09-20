"""Post-hoc selected-text control for the fixed/original-query confirmation."""
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
def run(source, episodes, reference, output):
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        if stop_requested(output):
            raise RuntimeError('Text control stop requested')
        checkpoint = resolve_checkpoint(source, verify=True)
        config = config_from_run(checkpoint)
        def save(path, value):
            ensure_free(output, config.train.min_free_disk_bytes)
            _atomic_text(path, json.dumps(value, indent=2)+'\n')
        prior = json.loads(reference.read_text())
        helper = Path(__file__).parents[2]/'scripts/evaluate_exact_detail.py'
        identity = {
            'checkpoint_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
            'episodes_sha256': file_sha256(episodes),
            'reference_sha256': file_sha256(reference),
            'script_sha256': file_sha256(__file__), 'max_new_tokens': 24,
            'selected_text_helper_sha256': file_sha256(helper),
        }
        for key in ('checkpoint_manifest_sha256', 'episodes_sha256'):
            if prior['inputs'][key] != identity[key]:
                raise ValueError('Text control differs from latent reference')
        if prior['inputs']['max_new_tokens'] != identity['max_new_tokens']:
            raise ValueError('Generation budget differs from reference')
        progress = output/'progress.json'
        rows = []
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved['inputs'] != identity:
                raise ValueError('Text control identity changed')
            rows = saved['rows']
        selected_text = runpy.run_path(str(helper))['selected_text']
        examples = [e for e in load_episodes(episodes) if e.task_family == 'multiuse/identifier']
        expected = [(e, c) for e in examples for c in ('text_required', 'none')]
        if len(rows) > len(expected):
            raise ValueError('Extra progress rows')
        for row, (episode, condition) in zip(rows, expected):
            if (row['episode'], row['condition'], row['answer']) != (episode.episode_id, condition, episode.answer):
                raise ValueError('Progress prefix differs')
            if row['exact_match'] != (row['prediction'].strip() == row['answer'].strip()):
                raise ValueError('Progress exact-match value differs')
        originals = {(r['episode'], r['condition']): r for r in prior['generation_rows']}
        for row in rows:
            if row['condition'] == 'none' and row['prediction'] != originals[(row['episode'], 'none')]['prediction']:
                raise ValueError('Saved no-memory prediction differs from reference')
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Text control invoked latent writer or compactor')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        scorer = FrozenScorer(agent)
        with autocast_context(config):
            for e, condition in expected[len(rows):]:
                if stop['signal'] is not None or stop_requested(output):
                    raise RuntimeError('Text control stopped; completed strings are resumable')
                context = selected_text(e) if condition == 'text_required' else ''
                tokens = agent.prompt_ids(e.query, context)
                prediction = scorer.generate(tokens, None, max_new_tokens=24)
                if condition == 'none' and prediction != originals[(e.episode_id, 'none')]['prediction']:
                    raise ValueError('No-memory prediction differs from frozen reference')
                rows.append({'episode': e.episode_id, 'environment': e.environment,
                             'condition': condition, 'answer': e.answer, 'prediction': prediction,
                             'exact_match': prediction.strip() == e.answer.strip(),
                             'prompt_tokens': tokens.numel(), 'source_text_bytes': len(context.encode())})
                save(progress, {'inputs': identity, 'rows': rows})
                if len(rows) % 32 == 0:
                    print(json.dumps({'completed': len(rows), 'total': len(expected)}), flush=True)
        summary = {c: {'n': sum(r['condition'] == c for r in rows),
                       'correct': sum(r['exact_match'] for r in rows if r['condition'] == c)}
                   for c in ('text_required', 'none')}
        save(output/'results.json', {'inputs': identity, 'summary': summary, 'rows': rows,
            'notice': 'Post-hoc diagnostic on the declared confirmation corpus. Same selected source information; '
                      'unequal text/latent token budgets. Source text is an inert control input, not a production '
                      'latent inference fallback. No candidate answers passed to generation.'})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'episodes', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.episodes, args.reference, args.output)
