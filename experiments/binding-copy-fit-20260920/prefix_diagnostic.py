"""Post-hoc teacher-forced suffix prediction under ambiguous/unique training prefixes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sdkb.archiving import ensure_free
from sdkb.checkpoints import _atomic_text, resolve_checkpoint, stop_on_signal
from sdkb.data import load_episodes
from sdkb.evaluation_adapter import load_frozen_agent
from sdkb.operations import run_lock, stop_requested
from sdkb.sessions import read_session
from sdkb.store import DiskStore
from sdkb.training import autocast_context, config_from_run
from sdkb.trajectories import file_sha256


@torch.no_grad()
def run(root):
    output = root/'prefix-diagnostic'
    output.mkdir(exist_ok=True)
    with run_lock(output, clear_stop=False), stop_on_signal() as stop:
        if stop_requested(output):
            raise RuntimeError('Prefix diagnostic stop requested')
        checkpoint = resolve_checkpoint(root/'copy', verify=True)
        reference = root/'confirmation/copy-training/results.json'
        prior = json.loads(reference.read_text())
        corpus, bank = root/'train.jsonl', reference.parent/'bank.sqlite'
        identity = {'checkpoint_manifest_sha256': file_sha256(checkpoint/'manifest.json'),
                    'episodes_sha256': file_sha256(corpus), 'bank_sha256': file_sha256(bank),
                    'reference_sha256': file_sha256(reference), 'script_sha256': file_sha256(__file__)}
        for key in ('checkpoint_manifest_sha256', 'episodes_sha256'):
            if prior['inputs'][key] != identity[key]:
                raise ValueError('Prefix diagnostic differs from training-set reference')
        if prior['bank_sha256'] != identity['bank_sha256']:
            raise ValueError('Training-set bank differs')
        config = config_from_run(checkpoint)
        torch.set_num_threads(config.train.threads)
        agent, _ = load_frozen_agent(config, checkpoint)
        agent.requires_grad_(False)
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Prefix diagnostic invoked a source writer')
        agent.produce = forbidden
        if agent.compactor is not None:
            agent.compactor.forward = forbidden
        episodes = load_episodes(corpus)
        if any(e.task_family != 'multiuse/identifier' for e in episodes):
            raise ValueError('Expected the declared identifier-only training subset')
        targets = {e.answer: agent.target_ids(e.answer).flatten().tolist() for e in episodes}
        baseline = {(r['episode'], r['condition']): r['target_mean_nll'] for r in prior['scores']['rows']}
        progress, rows = output/'progress.json', []
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved['inputs'] != identity:
                raise ValueError('Prefix progress identity differs')
            rows = saved['rows']
        if [r['episode'] for r in rows] != [e.episode_id for e in episodes[:len(rows)]]:
            raise ValueError('Prefix progress query sequence differs')
        def save(path, value):
            ensure_free(output, config.train.min_free_disk_bytes)
            _atomic_text(path, json.dumps(value, indent=2)+'\n')
        store = DiskStore(bank)
        with autocast_context(config):
            for e in episodes[len(rows):]:
                if stop['signal'] is not None or stop_requested(output):
                    raise RuntimeError('Prefix diagnostic stopped; completed queries are resumable')
                prompt = agent.prompt_ids(e.query)
                # Capture the memory before presenting any answer tokens to the decoder.
                memory = read_session(agent, store, prompt, namespace='global', generation='frozen-v1',
                                      query_time=e.query_time, oracle_ids=e.required_ids).memory
                target = agent.target_ids(e.answer)
                tokens = targets[e.answer]
                row = {'episode': e.episode_id, 'tokens': tokens, 'conditions': {}}
                row['prefix_candidates'] = [sum(candidate[:i] == tokens[:i] for candidate in targets.values())
                                            for i in range(len(tokens))]
                row['next_token_options'] = [len({candidate[i] for candidate in targets.values()
                    if len(candidate) > i and candidate[:i] == tokens[:i]}) for i in range(len(tokens))]
                row['categories'] = ['eos' if token == agent.tokenizer.eos_token_id else
                                     'unique_training_prefix' if row['prefix_candidates'][i] == 1 else
                                     'branching_training_prefix' if row['next_token_options'][i] > 1 else
                                     'shared_deterministic_prefix' for i, token in enumerate(tokens)]
                for condition, context in [('all', memory), ('none', None)]:
                    logits = agent.conditioned_logits(prompt, target, context)[0]
                    losses = F.cross_entropy(logits, target[0], reduction='none')
                    mean = float(F.cross_entropy(logits, target[0], reduction='sum'))/len(tokens)
                    if abs(mean-baseline[e.episode_id, condition]) > 1e-5:
                        raise ValueError('Per-token NLL differs from completed frozen reference')
                    row['conditions'][condition] = {'nll': losses.tolist(),
                        'correct': (logits.argmax(-1) == target[0]).tolist(),
                        'reference_mean_nll_absolute_error': abs(mean-baseline[e.episode_id, condition])}
                rows.append(row)
                save(progress, {'inputs': identity, 'rows': rows})
        grouped = defaultdict(lambda: defaultdict(list))
        for row in rows:
            for condition, values in row['conditions'].items():
                for i, category in enumerate(row['categories']):
                    grouped[condition][category].append((values['nll'][i], values['correct'][i]))
        summary = {condition: {category: {'tokens': len(values), 'mean_nll': sum(v[0] for v in values)/len(values),
                    'correct': sum(v[1] for v in values)} for category, values in categories.items()}
                   for condition, categories in grouped.items()}
        save(output/'results.json', {'inputs': identity, 'summary': summary, 'rows': rows,
            'notice': 'Post-hoc teacher-forced training-set diagnostic, not free generation or held-out evidence. '
                      'Prefix uniqueness is computed over 64 training answers only, never supplied to the model. '
                      'Queries/reads exclude target tokens; the decoder sees only the ordinary causal answer prefix.'})
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
