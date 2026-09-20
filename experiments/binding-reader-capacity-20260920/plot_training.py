"""Plot a consistent prefix of training logs; capability results remain separate."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    inputs = json.loads((root/'inputs.json').read_text())
    output = root/'analysis'
    output.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.6))
    report = {'window_updates': 100, 'arms': {}, 'notice': 'Training fit only; ongoing logs may have unequal prefixes. No held-out or throughput claim.'}
    for (name, _digest), color in zip(inputs['configs'].items(), ('#0072B2', '#D55E00'), strict=True):
        # A concurrent append may end halfway through its final JSON row.
        raw = (root/name/'metrics.jsonl').read_bytes()
        raw = raw[:raw.rfind(b'\n')+1]
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) < 100 or [r['step'] for r in rows] != list(range(1, len(rows)+1)):
            raise ValueError('Expected a complete consecutive prefix of at least 100 updates')
        values = np.asarray([r['nll'] for r in rows])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite training loss')
        axis.plot(np.arange(100, len(values)+1), np.convolve(values, np.ones(100)/100, mode='valid'),
                  color=color, label=name.replace('width-', '')+'-wide reader', linewidth=1.8)
        report['arms'][name] = {'complete_updates': len(rows), 'metrics_prefix_sha256': hashlib.sha256(raw).hexdigest(),
            'first_100_update_mean_nll': float(values[:100].mean()), 'last_100_update_mean_nll': float(values[-100:].mean())}
    axis.set(xlabel='Optimizer updates', ylabel='Mean target NLL (100-update window)',
             title='Reset-reader capacity comparison: training fit', ylim=(0, None))
    axis.legend(frameon=False)
    axis.grid(alpha=.2)
    figure.text(.5, .025, 'Teacher-forced training loss, one seed, same source and task schedule.\n'
                'Different reader sizes and compute; exact recall and rule retention require held-out evaluation.',
                ha='center', fontsize=8.5)
    figure.tight_layout(rect=(0, .09, 1, 1))
    for extension in ('png', 'svg'):
        figure.savefig(output/f'training-nll.{extension}', dpi=160)
    plt.close(figure)
    (output/'training-nll.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    plot(parser.parse_args().root)
