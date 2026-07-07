import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(args):
    widths, flag, inp, cls, flag_dr, inp_dr, cls_dr = [], [], [], [], [], [], []
    for w in args.widths:
        tag = f'{w:g}' if isinstance(w, float) and w != int(w) else f'{int(w)}'
        m = Path(args.root) / f'{args.prefix}{tag}' / 'metrics.json'
        if not m.exists():
            print(f'missing {m}, skipping {args.prefix}={w}')
            continue
        d = json.loads(m.read_text())
        if 'flagged' not in d or 'inpainted' not in d:
            print(f'w={w}: no flagged/inpainted metrics, skipping')
            continue
        widths.append(w)
        flag.append(d['flagged'].get('rmse_vs_clean', np.nan))
        inp.append(d['inpainted'].get('rmse_vs_clean', np.nan))
        cls.append(d.get('classical', {}).get('rmse_vs_clean', np.nan))
        flag_dr.append(d['flagged'].get('dr', np.nan))
        inp_dr.append(d['inpainted'].get('dr', np.nan))
        cls_dr.append(d.get('classical', {}).get('dr', np.nan))
    if not widths:
        raise SystemExit('no metrics found')
    widths = np.array(widths, float)
    flag, inp, cls = np.array(flag), np.array(inp), np.array(cls)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax.plot(widths, flag * 1e6, 'o-', color='0.5', label='flagged (discard RFI)')
    ax.plot(widths, inp * 1e6, 'o-', color='tab:red', label='inpainted (diffusion model)')
    if np.isfinite(cls).any():
        ax.plot(widths, cls * 1e6, 'o-', color='tab:orange', label='DPSS classical fill')
    below = inp < flag
    if below.any():
        ax.fill_between(widths, 0, ax.get_ylim()[1], where=below, color='tab:green', alpha=0.08)
    # crossover: first width where inpaint stops beating flagging
    cross = None
    for i in range(len(widths) - 1):
        if inp[i] <= flag[i] and inp[i + 1] > flag[i + 1]:
            cross = 0.5 * (widths[i] + widths[i + 1]); break
    if cross:
        ax.axvline(cross, color='k', ls=':', lw=1)
        ax.text(cross, ax.get_ylim()[1] * 0.95, f' crossover ~{cross:.0f} ch', fontsize=8, va='top')
    def setx(a):
        if args.logx:
            a.set_xscale('log', base=2)
        a.set_xticks(widths); a.set_xticklabels([f'{w:g}' for w in widths])
        a.set_xlabel(args.xlabel)
    setx(ax); ax.set_ylabel('image RMSE vs clean (uJy-scale, x1e6)')
    ax.set_title(f'Continuum fidelity vs {args.xlabel}: inpaint vs flag'); ax.legend()

    ax2.plot(widths, flag_dr, 'o-', color='0.5', label='flagged')
    ax2.plot(widths, inp_dr, 'o-', color='tab:red', label='inpainted')
    if np.isfinite(cls_dr).any():
        ax2.plot(widths, cls_dr, 'o-', color='tab:orange', label='DPSS classical')
    setx(ax2); ax2.set_ylabel('dynamic range (higher = better)')
    ax2.set_title(f'Dynamic range vs {args.xlabel}'); ax2.legend()

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'{"x":>8}{"flag RMSE":>14}{"inp RMSE":>14}{"dpss RMSE":>14}{"inp<flag":>10}')
    for i, w in enumerate(widths):
        print(f'{w:>8g}{flag[i]:>14.3e}{inp[i]:>14.3e}{cls[i]:>14.3e}{str(inp[i] < flag[i]):>10}')
    print(f'saved -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dir containing <prefix><val>/metrics.json subdirs')
    ap.add_argument('--widths', nargs='+', type=float, required=True, help='x values (widths or noise scales)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', default='w', help="subdir prefix: 'w' for width, 'n' for noise")
    ap.add_argument('--xlabel', default='RFI band width (native channels)')
    ap.add_argument('--logx', action='store_true', help='log2 x-axis (widths); omit for linear (noise)')
    main(ap.parse_args())
