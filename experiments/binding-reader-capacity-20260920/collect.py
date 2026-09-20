"""Collect all frozen capacity-study confirmations after verifying coverage."""
import argparse
import json
from pathlib import Path
import runpy

from sdkb.checkpoints import resolve_checkpoint
from sdkb.operations import atomic_json
from sdkb.trajectories import file_sha256


def collect(root, output):
    code = Path(__file__).parents[2]
    helper = code/'experiments/binding-endpoint-freshness-20260920/summarize.py'
    confirmation = runpy.run_path(str(helper))['confirmation']
    inputs = json.loads((root/'inputs.json').read_text())
    episodes = Path(inputs['heldout'])
    if file_sha256(episodes) != inputs['heldout_sha256']:
        raise ValueError('Held-out corpus changed')
    text_inputs = json.loads((root/'text-controls.json').read_text())
    if (text_inputs['episodes_sha256'] != inputs['heldout_sha256']
            or set(text_inputs['arms']) != set(inputs['configs'])):
        raise ValueError('Supplemental text declaration differs')
    reports = {}
    for name in inputs['configs']:
        checkpoint = resolve_checkpoint(root/name, verify=True)
        if json.loads((checkpoint/'manifest.json').read_text())['step'] != inputs['steps']:
            raise ValueError('Incomplete training endpoint')
        report, rows = confirmation(root/'confirmation'/name/'results.json', checkpoint, episodes)
        path = root/'confirmation'/(name+'-text')/'results.json'
        text = json.loads(path.read_text())
        if text['inputs']['reference_sha256'] != report['results_sha256']:
            raise ValueError('Text reference differs')
        original = {r['episode']: r['prediction'] for r in rows
                    if r['task_family'] == 'multiuse/identifier' and r['condition'] == 'none'}
        repeated = {r['episode']: r['prediction'] for r in text['rows'] if r['condition'] == 'none'}
        if original != repeated or len(text['rows']) != 2*len(original):
            raise ValueError('Incomplete text control or changed no-memory strings')
        report['selected_text'] = {'inputs': text['inputs'], 'summary': text['summary'],
                                  'results_sha256': file_sha256(path), 'no_memory_strings_match': True}
        reports[name] = report
    atomic_json(output, {'arms': reports, 'inputs_sha256': file_sha256(root/'inputs.json'),
        'text_controls_sha256': file_sha256(root/'text-controls.json'),
        'collector_sha256': file_sha256(__file__), 'coverage_helper_sha256': file_sha256(helper),
        'notice': inputs['notice']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
