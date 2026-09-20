"""Collect complete freshness confirmations with identity and row-coverage checks."""
import argparse
from collections import Counter
import json
from pathlib import Path

from sdkb.checkpoints import resolve_checkpoint
from sdkb.data import load_episodes
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


CONDITIONS = {'all', 'none', 'zero_values', 'cf_identifier', 'cf_permission', 'cf_restoration'}


def confirmation(path, checkpoint, episodes):
    result = json.loads(path.read_text())
    if (result['inputs']['checkpoint_manifest_sha256'] != file_sha256(checkpoint/'manifest.json')
            or result['inputs']['episodes_sha256'] != file_sha256(episodes)
            or result['inputs']['max_new_tokens'] != 24
            or result['inputs']['compact_method'] != 'raw'):
        raise ValueError(f'Confirmation identity mismatch: {path}')
    expected = {(e.episode_id, c) for e in load_episodes(episodes) for c in CONDITIONS}
    rows = result['generation_rows']
    keys = [(r['episode'], r['condition']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f'Incomplete or repeated confirmation rows: {path}')
    summary = {}
    for row in rows:
        if row['exact_match'] != (row['answer'] == row['prediction']):
            raise ValueError('Incorrect stored exact-match flag')
        counts = summary.setdefault(row['task_family'], {}).setdefault(
            row['condition'], {'n': 0, 'correct': 0})
        counts['n'] += 1
        counts['correct'] += row['exact_match']
    if summary != result['generation_summary']:
        raise ValueError('Reported summary differs from complete generation rows')
    nll = {}
    for family in summary:
        nll[family] = {}
        for condition in CONDITIONS:
            group = [r for r in result['scores']['rows']
                     if r['task_family'] == family and r['condition'] == condition]
            if (len(group) != summary[family][condition]['n'] or
                    {(r['episode'], r['condition']) for r in group} != {
                    (r['episode'], r['condition']) for r in rows
                    if r['task_family'] == family and r['condition'] == condition}):
                raise ValueError('Teacher scores do not cover generation questions')
            nll[family][condition] = sum(r['target_mean_nll'] for r in group)/len(group)
    return {'inputs': result['inputs'], 'generation_summary': summary,
            'generation_counterfactuals': result['generation_counterfactuals'],
            'mean_question_target_nll': nll, 'bank_sha256': result['bank_sha256'],
            'results_sha256': file_sha256(path)}, rows


def summarize(root, output):
    inputs = json.loads((root/'inputs.json').read_text())
    fit = json.loads((root/'training-diagnostic/inputs.json').read_text())
    if file_sha256(root/'heldout.jsonl') != inputs['heldout_sha256']:
        raise ValueError('Held-out input changed')
    fit_episodes = root/'training-diagnostic/episodes.jsonl'
    if file_sha256(fit_episodes) != fit['episodes_sha256']:
        raise ValueError('Supplemental input changed')
    result = {'heldout': {}, 'training_diagnostic': {}, 'selected_text': {}}
    checkpoints = {'source-heldout': resolve_checkpoint(inputs['source'])}
    checkpoints.update({f'worlds-{s}': resolve_checkpoint(root/f'worlds-{s}') for s in fit['arms']})
    for label, checkpoint in checkpoints.items():
        if label != 'source-heldout' and json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
            raise ValueError('Training is incomplete')
        path = root/'confirmation'/label/'results.json'
        report, rows = confirmation(path, checkpoint, root/'heldout.jsonl')
        result['heldout'][label] = report
        text_label = 'source-text' if label == 'source-heldout' else label+'-text'
        text_path = ((root/text_label if label == 'source-heldout'
                      else root/'confirmation'/text_label)/'results.json')
        text = json.loads(text_path.read_text())
        if text['inputs']['reference_sha256'] != report['results_sha256']:
            raise ValueError('Text control belongs to another latent reference')
        original_none = {r['episode']: r['prediction'] for r in rows
                         if r['task_family'] == 'multiuse/identifier' and r['condition'] == 'none'}
        text_none = {r['episode']: r['prediction'] for r in text['rows'] if r['condition'] == 'none'}
        if original_none != text_none or len(text['rows']) != 2*len(original_none):
            raise ValueError('Incomplete text control or changed no-memory predictions')
        result['selected_text'][label] = {'inputs': text['inputs'], 'summary': text['summary'],
            'results_sha256': file_sha256(text_path), 'all_no_memory_predictions_match_reference': True}
        if label == 'source-heldout':
            continue
        size = label.removeprefix('worlds-')
        report, rows = confirmation(root/'confirmation'/(label+'-training')/'results.json',
                                    checkpoint, fit_episodes)
        exposure = {r['episode']: r['identifier_target_exposures'] for r in fit['arms'][size]['rows']}
        buckets = {}
        for row in rows:
            count = exposure[row['episode']]
            bucket = 'zero' if count == 0 else 'one' if count == 1 else 'multiple'
            counts = buckets.setdefault(bucket, {}).setdefault(row['condition'], Counter())
            counts['n'] += 1
            counts['correct'] += row['exact_match']
        report['by_endpoint_target_exposure'] = buckets
        result['training_diagnostic'][label] = report
    result['inputs_sha256'] = file_sha256(root/'inputs.json')
    result['supplemental_inputs_sha256'] = file_sha256(root/'training-diagnostic/inputs.json')
    result['script_sha256'] = file_sha256(__file__)
    result['notice'] = ('One seed, oracle-selected stored payloads, fixed held-out questions. '
        'The training subset is supplemental, with target-exposure strata rather than claims of '
        'unseen source text. Text and latent token budgets differ. Mean question target NLL is '
        'teacher-forced and separate from exact generation and correct counterfactual pairs.')
    atomic_json(output, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
