"""Plot matched complete training prefixes, separating task and alignment losses."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    loaded = {}
    for name in ('control', 'alignment'):
        raw = (root/name/'metrics.jsonl').read_bytes()
        lines = raw[:raw.rfind(b'\n')+1].splitlines(keepends=True)
        rows = [json.loads(line) for line in lines]
        if [r['step'] for r in rows] != list(range(1, len(rows)+1)):
            raise ValueError('Nonconsecutive metric prefix')
        loaded[name] = lines, rows
    count = min(len(rows) for _, rows in loaded.values())
    window = 100
    if count < window:
        raise ValueError('At least 100 complete matched updates required')
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    report = dict(common_updates=count, smoothing_window=window, arms={},
                  notice='Training diagnostics only. Alignment loss is not measured in the control; its logged zero is not a comparison value. Same update count is not equal compute.')
    for name, color in [('control', '#0072B2'), ('alignment', '#D55E00')]:
        lines, rows = loaded[name]
        rows = rows[:count]
        report['arms'][name] = dict(metrics_prefix_sha256=hashlib.sha256(b''.join(lines[:count])).hexdigest())
        for metric, axis in [('nll', axes[0]), ('oracle_alignment_loss', axes[1])]:
            if name == 'control' and metric == 'oracle_alignment_loss':
                continue
            values = np.asarray([r[metric] for r in rows])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite metric')
            smooth = np.convolve(values, np.ones(window)/window, mode='valid')
            axis.plot(np.arange(window, count+1), smooth, color=color, label=name, linewidth=1.8)
            report['arms'][name][metric] = dict(first_100=float(values[:window].mean()),
                                               last_100=float(values[-window:].mean()))
    axes[0].set_title('Teacher-forced target NLL')
    axes[1].set_title('Detached text-state alignment loss')
    for axis in axes:
        axis.set_xlabel('Optimizer updates (100-update smoothing)')
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    figure.text(.5, .025, 'Same source, data and sampler; one seed. Training losses do not establish exact recall or agent success.',
                ha='center', fontsize=8.5)
    figure.tight_layout(rect=(0, .07, 1, 1))
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    for extension in ('png', 'svg'):
        figure.savefig(output/f'training-losses.{extension}', dpi=160)
    plt.close(figure)
    (output/'training-losses.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    plot(parser.parse_args().root)
