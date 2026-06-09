import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def green_overlay(mask_2d):
    rgba = np.zeros((*mask_2d.T.shape, 4), dtype=np.float32)
    rgba[mask_2d.T > 0] = [0.0, 1.0, 0.2, 0.85]
    return rgba


def plot_npz(path, out_dir, n_show):
    d = np.load(path)
    clean, corrupted, mask, pred = d['clean'], d['corrupted'], d['mask'], d['pred']
    clean = clean.squeeze(1) if clean.ndim == 4 else clean
    corrupted = corrupted.squeeze(1) if corrupted.ndim == 4 else corrupted
    mask = mask.squeeze(1) if mask.ndim == 4 else mask
    pred = pred.squeeze(1) if pred.ndim == 4 else pred

    n = min(n_show, clean.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["clean (truth)", "corrupted + mask", "predicted", "error in mask"]
    for i in range(n):
        vmin = np.percentile(clean[i], 1)
        vmax = np.percentile(clean[i], 99)
        m = mask[i]

        axes[i, 0].imshow(clean[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="plasma")
        axes[i, 1].imshow(corrupted[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="plasma")
        axes[i, 1].imshow(green_overlay(m), aspect="auto", origin="lower")
        axes[i, 2].imshow(pred[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="plasma")

        err = np.abs(pred[i] - clean[i])
        err[m == 0] = np.nan
        axes[i, 3].imshow(err.T, aspect="auto", origin="lower", cmap="magma")

        mae_mask = err[~np.isnan(err)].mean() if np.isfinite(err).any() else 0.0
        axes[i, 0].set_ylabel(f"sample {i}\nmask MAE={mae_mask:.4f}", fontsize=8)
        if i == 0:
            for j, t in enumerate(titles):
                axes[i, j].set_title(t, fontsize=9)
        for ax in axes[i]:
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = out_dir / (Path(path).stem + ".png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(args.input)) if any(c in args.input for c in '*?[') else [args.input]
    if not paths:
        raise SystemExit(f"no npz matched {args.input}")
    for p in paths:
        plot_npz(p, out_dir, args.n_show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-show", type=int, default=6, dest="n_show")
    main(ap.parse_args())
