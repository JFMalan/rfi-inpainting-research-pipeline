import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def overlay(mask2d, rgba):
    out = np.zeros((*mask2d.T.shape, 4), np.float32)
    out[mask2d.T > 0] = rgba
    return out


def main(args):
    hf = h5py.File(args.h5, 'r')
    pr = np.load(args.preds)['preds']
    n = min(hf['corrupted'].shape[0], pr.shape[0])
    has_target = 'amp_target' in hf
    has_fpatch = 'freq_min_patch' in hf

    frac = hf['mask'][:n].astype(np.float32).mean(axis=(1, 2))
    pool = np.where((frac >= args.min_flag_frac) & (frac < args.max_flag_frac))[0]
    rng = np.random.default_rng(args.seed)
    if len(pool) > args.n_show:
        pool = np.sort(rng.choice(pool, args.n_show, replace=False))
    print(f"{len(pool)} units (mask frac {args.min_flag_frac}-{args.max_flag_frac}, "
          f"pool of {n})", flush=True)

    ncols = 5 if has_target else 4
    titles = ["observed (RFI)", "injected mask", "model fill",
              "noisy truth (no RFI)"] + (["clean target"] if has_target else [])
    fig, axes = plt.subplots(len(pool), ncols, figsize=(3.7 * ncols, 3.2 * len(pool)),
                             squeeze=False)
    for r, u in enumerate(pool):
        obs = hf['corrupted'][u].astype(np.float32)
        mask = hf['mask'][u].astype(np.float32)
        clean = hf['clean'][u].astype(np.float32)
        fill = np.where(mask > 0.5, pr[u, 0], obs)
        imgs = [obs, obs, fill, clean]
        if has_target:
            tgt = hf['amp_target'][u].astype(np.float32)
            imgs.append(tgt)
        if has_fpatch:
            fmin = float(hf['freq_min_patch'][u]); fmax = float(hf['freq_max_patch'][u])
        else:
            fmin = float(hf.attrs['freq_min_mhz']); fmax = float(hf.attrs['freq_max_mhz'])
        vmin, vmax = np.percentile(clean, 1), np.percentile(clean, 99)
        ext = [0, obs.shape[0], fmin, fmax]
        for j, img in enumerate(imgs):
            axes[r, j].imshow(img.T, aspect='auto', origin='lower', extent=ext,
                              vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 1].imshow(overlay(mask, [0.0, 1.0, 0.2, 0.85]),
                          aspect='auto', origin='lower', extent=ext)
        hole = mask > 0.5
        mae_n = np.abs(pr[u, 0] - clean)[hole].mean()
        lb = f"unit {u}  mask={mask.mean():.2f}\nMAE vs noisy {mae_n:.3f}"
        if has_target:
            lb += f"\nMAE vs clean {np.abs(pr[u, 0] - tgt)[hole].mean():.3f}"
        axes[r, 0].set_ylabel(lb + "\nFreq (MHz)", fontsize=7)
        if r == 0:
            for j, t in enumerate(titles):
                axes[r, j].set_title(t, fontsize=9)
        for ax in axes[r]:
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Time bin", fontsize=7)
    hf.close()

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"-> {out}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-show', type=int, default=8, dest='n_show')
    ap.add_argument('--min-flag-frac', type=float, default=0.1, dest='min_flag_frac')
    ap.add_argument('--max-flag-frac', type=float, default=0.6, dest='max_flag_frac')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
