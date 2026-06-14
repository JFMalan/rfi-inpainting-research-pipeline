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


def _amp_phase(x):
    # x: (N, C, H, W). C>=3 -> amplitude ch0, phase = atan2(sin ch2, cos ch1).
    if x.ndim == 4 and x.shape[1] >= 3:
        amp = x[:, 0]
        ph = np.arctan2(x[:, 2], x[:, 1])
        return amp, ph
    amp = x.squeeze(1) if x.ndim == 4 else x
    return amp, None


def plot_npz(path, out_dir, n_show):
    d = np.load(path)
    clean_a, clean_p = _amp_phase(d['clean'])
    corr_a, _ = _amp_phase(d['corrupted'])
    pred_a, pred_p = _amp_phase(d['pred'])
    mask = d['mask']
    mask = mask.squeeze(1) if mask.ndim == 4 else mask
    fmin = d['fmin'] if 'fmin' in d else None
    fmax = d['fmax'] if 'fmax' in d else None
    has_phase = clean_p is not None

    n = min(n_show, clean_a.shape[0])
    n_time = clean_a.shape[1]
    ncols = 6 if has_phase else 4
    fig, axes = plt.subplots(n, ncols, figsize=(3.7 * ncols, 3.4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    titles = ["clean amp", "corrupted + mask", "predicted amp", "amp error in mask"]
    if has_phase:
        titles += ["clean phase", "predicted phase"]
    for i in range(n):
        vmin = np.percentile(clean_a[i], 1)
        vmax = np.percentile(clean_a[i], 99)
        m = mask[i]
        if fmin is not None:
            ext = [0, n_time, float(fmin[i]), float(fmax[i])]
            ylab = "Freq (MHz)"
        else:
            ext = None
            ylab = "Freq channel"

        axes[i, 0].imshow(clean_a[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                          extent=ext, cmap="plasma")
        axes[i, 1].imshow(corr_a[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                          extent=ext, cmap="plasma")
        axes[i, 1].imshow(green_overlay(m), aspect="auto", origin="lower", extent=ext)
        axes[i, 2].imshow(pred_a[i].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax,
                          extent=ext, cmap="plasma")

        err = np.abs(pred_a[i] - clean_a[i])
        err[m == 0] = np.nan
        # fixed error scale (0 .. data dynamic range) so magnitude is honest, not
        # per-panel autoscaled (which makes tiny errors look catastrophic on flat patches)
        axes[i, 3].imshow(err.T, aspect="auto", origin="lower", extent=ext, cmap="magma",
                          vmin=0.0, vmax=(vmax - vmin))

        if has_phase:
            axes[i, 4].imshow(clean_p[i].T, aspect="auto", origin="lower", vmin=-np.pi, vmax=np.pi,
                              extent=ext, cmap="twilight")
            axes[i, 5].imshow(pred_p[i].T, aspect="auto", origin="lower", vmin=-np.pi, vmax=np.pi,
                              extent=ext, cmap="twilight")

        mae_mask = err[~np.isnan(err)].mean() if np.isfinite(err).any() else 0.0
        in_hole = pred_a[i][m > 0]
        axes[i, 0].set_ylabel(
            f"sample {i}\n{ylab}\nmaskMAE={mae_mask:.4f}\n"
            f"clean[{vmin:.2f},{vmax:.2f}] pred_hole[{in_hole.min():.2f},{in_hole.max():.2f}]",
            fontsize=7)
        if i == 0:
            for j, t in enumerate(titles):
                axes[i, j].set_title(t, fontsize=9)
        for ax in axes[i]:
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Time bin", fontsize=7)

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
