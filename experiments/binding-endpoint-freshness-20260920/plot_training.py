"""Plot completed training fit separately from held-out capability evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sdkb.trajectories import file_sha256


def plot(root):
    inputs = json.loads((root/'inputs.json').read_text())
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.6))
    reports = {}
    window = 100
    for size, color in [('256', '#0072B2'), ('8192', '#D55E00')]:
        path = root/f'worlds-{size}'/'metrics.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [r['step'] for r in rows] != list(range(1, inputs['steps']+1)):
            raise ValueError('Expected one complete row per declared optimizer update')
        values = np.asarray([r['nll'] for r in rows])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite target loss')
        rolling = np.convolve(values, np.ones(window)/window, mode='valid')
        axis.plot(np.arange(window, len(values)+1), rolling, color=color,
                  label=f'{int(size):,} worlds', linewidth=1.8)
        reports[size] = {'metrics_sha256': file_sha256(path),
                         'first_100_update_mean_nll': float(values[:window].mean()),
                         'last_100_update_mean_nll': float(values[-window:].mean())}
    axis.set(xlabel='Optimizer updates', ylabel='Mean target NLL (100-update window)',
             title='Endpoint freshness: training fit', ylim=(0, None))
    axis.legend(frameon=False)
    axis.grid(alpha=.2)
    figure.text(.5, .025, 'Teacher-forced training loss, one seed. Same task schedule; different data streams.\n'
                'Exact generation, counterfactual recall and rule retention are separate evaluations.',
                ha='center', fontsize=8.5)
    figure.tight_layout(rect=(0, .09, 1, 1))
    for extension in ('png', 'svg'):
        figure.savefig(output/f'training-nll.{extension}', dpi=160)
    plt.close(figure)
    report = {'window_updates': window, 'arms': reports,
              'script_sha256': file_sha256(__file__),
              'notice': 'Descriptive training fit only. No independent-seed uncertainty or held-out claim.'}
    (output/'training-nll.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    plot(parser.parse_args().root)
