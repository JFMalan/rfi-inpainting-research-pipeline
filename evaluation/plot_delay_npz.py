import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

META = {'fg_bins', 'n', 'floors'}
LABELS = {'truth': 'true good data', 'flagged': 'flagged (zero-fill)',
          'dpss': 'DPSS (classical)', 'gpr': 'GPR (classical)'}
COLORS = {'truth': 'k', 'flagged': '0.6', 'dpss': 'tab:orange', 'gpr': 'tab:green'}
MODEL_COLORS = ['tab:red', 'tab:purple', 'tab:blue', 'tab:brown']


def main(args):
    d = np.load(args.npz)
    fg_bins = int(d['fg_bins']) if 'fg_bins' in d else args.fg_bins
    variants = [k for k in d.files if k not in META and d[k].ndim == 1]
    sz = len(d[variants[0]])
    center = sz // 2
    tau = np.arange(sz) - center
    hi = np.abs(tau) > fg_bins
    eps = 1e-30
    truth = d['truth']
    w = truth + eps

    def wlogrmse(k):
        return float(np.sqrt((w * (np.log10(d[k] + eps) - np.log10(truth + eps)) ** 2).sum() / w.sum()))

    def hiratio(k):
        return float(d[k][hi].sum() / max(truth[hi].sum(), eps))

    order = [v for v in ['truth', 'flagged', 'dpss', 'gpr'] if v in variants]
    models = sorted(v for v in variants if v.startswith('model'))
    order += models

    fig, ax = plt.subplots(figsize=(8, 5))
    mi = 0
    for k in order:
        if k.startswith('model'):
            nf = k.replace('model_nf', '')
            lab = f'model (noise_floor={nf})' if nf != 'none' else 'model (no texture)'
            color = MODEL_COLORS[mi % len(MODEL_COLORS)]; mi += 1
        else:
            lab = LABELS.get(k, k)
            color = COLORS.get(k)
        if k != 'truth':
            lab += f'   wlogP-RMSE={wlogrmse(k):.3f}, hi-ratio={hiratio(k):.2f}'
        ax.semilogy(tau, d[k] + eps, label=lab,
                    color=color, lw=2.0 if k == 'truth' else 1.4,
                    ls='--' if k == 'truth' else '-', zorder=3 if k.startswith('model') else 2)
    ax.axvspan(-fg_bins, fg_bins, color='0.9', zorder=0)
    ax.text(0, ax.get_ylim()[1], ' foreground\n (low delay)', va='top', ha='center', fontsize=7, color='0.4')
    ax.set_xlabel('delay bin (relative to zero delay)')
    ax.set_ylabel('delay-space power (per-baseline mean)')
    n = int(d['n']) if 'n' in d else 0
    title = args.title or f'Real held-out delay-space recovery ({n} fake-hole tiles vs true good data)'
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    if args.gap_ci:
        ax.text(0.02, 0.02, args.gap_ci, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', fc='white', ec='0.7'))
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'variants: {order}')
    for k in order:
        if k != 'truth':
            print(f'  {k:<16} wlogP-RMSE={wlogrmse(k):.4f}  hi-ratio={hiratio(k):.3f}')
    print(f'saved -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='fakehole_delay_eval output (e.g. fakehole_delay_finetune.npz)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default=None)
    ap.add_argument('--fg-bins', type=int, default=20, dest='fg_bins')
    ap.add_argument('--gap-ci', default=None, dest='gap_ci',
                    help='optional annotation text, e.g. "model vs DPSS: +0.031, 95% CI [+0.003, +0.077]"')
    main(ap.parse_args())
