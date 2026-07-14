import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def overlay(mask2d, rgba):
    out = np.zeros((*mask2d.T.shape, 4), np.float32)
    out[mask2d.T > 0] = rgba
    return out


def main(args):
    rows = []
    for path in args.npz:
        d = np.load(path)
        tag = Path(path).stem.replace('sample_', '')
        b = d['obs'].shape[0]
        if args.max_per_file:
            b = min(b, args.max_per_file)
        for i in range(b):
            rows.append((tag, i, d['obs'][i], d['real_flags'][i, 0],
                         d['fake_mask'][i, 0], d['pred'][i]))

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.1 * n), squeeze=False)
    titles = ["observed amp", "masks (red=RFI, green=fake)",
              "fill (pred in holes)", "raw prediction"]
    for r, (tag, i, obs, flags, fake, pred) in enumerate(rows):
        amp = obs[0]
        hidden = np.clip(flags + fake, 0, 1)
        fill = np.where(hidden > 0.5, pred[0], amp)
        trust = hidden < 0.5
        src = amp[trust] if trust.any() else amp
        vmin, vmax = np.percentile(src, 1), np.percentile(src, 99)
        for j, img in enumerate((amp, amp, fill, pred[0])):
            axes[r, j].imshow(img.T, aspect='auto', origin='lower',
                              vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 1].imshow(overlay(flags, [1.0, 0.1, 0.1, 0.8]), aspect='auto', origin='lower')
        axes[r, 1].imshow(overlay(fake, [0.0, 1.0, 0.2, 0.8]), aspect='auto', origin='lower')
        axes[r, 0].set_ylabel(f"{tag} sample {i}\nflag={flags.mean():.2f} fake={fake.mean():.2f}\n"
                              f"scale[{vmin:.2f},{vmax:.2f}]", fontsize=7)
        if r == 0:
            for j, t in enumerate(titles):
                axes[r, j].set_title(t, fontsize=9)
        for ax in axes[r]:
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Time bin", fontsize=7)

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"-> {out}  ({n} rows)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-per-file', type=int, default=None, dest='max_per_file')
    main(ap.parse_args())
