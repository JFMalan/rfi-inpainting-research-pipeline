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

    def smooth(y, w):
        if w <= 1:
            return y
        k = np.ones(w) / w
        return np.convolve(y, k, mode='same')

    def style(k):
        if k.startswith('model'):
            nf = k.replace('model_nf', '')
            lab = f'model (noise_floor={nf})' if nf != 'none' else 'model (no texture)'
            return lab, MODEL_COLORS[models.index(k) % len(MODEL_COLORS)]
        return LABELS.get(k, k), COLORS.get(k)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 2]})
    for k in order:
        lab, color = style(k)
        if k != 'truth':
            lab += f'   wlogP-RMSE={wlogrmse(k):.3f}, hi-ratio={hiratio(k):.2f}'
        ax.semilogy(tau, d[k] + eps, label=lab, color=color, lw=2.0 if k == 'truth' else 1.4,
                    ls='--' if k == 'truth' else '-', zorder=3 if k.startswith('model') else 2)
    ax.axvspan(-fg_bins, fg_bins, color='0.9', zorder=0)
    ax.set_ylabel('delay power (per-baseline mean)')
    n = int(d['n']) if 'n' in d else 0
    ax.set_title(args.title or f'Real held-out delay-space recovery ({n} tiles vs true good data)', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')

    # ratio-to-truth panel: truth = 1.0; closest to 1 across the tail wins (this is what hi-ratio measures)
    for k in order:
        if k == 'truth':
            continue
        lab, color = style(k)
        ax2.plot(tau, smooth(d[k] / (truth + eps), args.smooth), color=color, lw=1.4,
                 zorder=3 if k.startswith('model') else 2)
    ax2.axhline(1.0, color='k', ls='--', lw=1.5, zorder=4)
    ax2.axvspan(-fg_bins, fg_bins, color='0.9', zorder=0)
    ax2.set_ylim(0, 1.8)
    ax2.set_xlabel('delay bin (relative to zero delay)')
    ax2.set_ylabel('power / truth\n(1.0 = perfect)')
    ax2.text(0.99, 0.05, 'below 1 = under-recovers   above 1 = fabricates',
             transform=ax2.transAxes, ha='right', fontsize=7, color='0.4')
    if args.gap_ci:
        ax2.text(0.02, 0.92, args.gap_ci, transform=ax2.transAxes, fontsize=8, va='top',
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
    ap.add_argument('--smooth', type=int, default=7, help='moving-average width for the ratio panel')
    main(ap.parse_args())
