import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(args):
    d = np.load(args.input)
    clean = d['clean'][:, 0]; mask = d['mask'][:, 0]
    pred_keys = [k for k in d.files if k.startswith('pred_')]
    preds = {k.replace('pred_', ''): d[k][:, 0] for k in pred_keys}
    n = min(args.n, clean.shape[0])
    first = list(preds)[0]

    cols = ['clean'] + list(preds) + [f'err({first})']
    fig, axes = plt.subplots(n, len(cols), figsize=(3.2 * len(cols), 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i in range(n):
        vmin, vmax = np.percentile(clean[i], 1), np.percentile(clean[i], 99)
        m = mask[i] > 0
        axes[i, 0].imshow(clean[i].T, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='plasma')
        for j, (name, p) in enumerate(preds.items(), 1):
            axes[i, j].imshow(p[i].T, origin='lower', aspect='auto', vmin=vmin, vmax=vmax, cmap='plasma')
        err = np.abs(preds[first][i] - clean[i]); err[~m] = np.nan
        axes[i, -1].imshow(err.T, origin='lower', aspect='auto', vmin=0, vmax=(vmax - vmin), cmap='magma')
        mm = (np.abs(preds[first][i] - clean[i]))[m].mean()
        axes[i, 0].set_ylabel(f"s{i}  MAE={mm:.4f}", fontsize=8)
        if i == 0:
            for j, c in enumerate(cols):
                axes[i, j].set_title(c, fontsize=9)
    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=120, bbox_inches='tight'); plt.close()
    print(f"-> {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=6)
    main(ap.parse_args())
