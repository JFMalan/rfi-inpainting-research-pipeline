import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEFD_MHZ = np.array([856, 900, 950, 1000, 1100, 1280, 1400, 1450, 1550, 1600, 1650, 1712])
SEFD_JY = np.array([560, 510, 450, 420, 390, 390, 400, 420, 450, 470, 500, 560])
DNU = 856e6 / 1024
DT = 8.0


def sigma_of(scale):
    sefd = np.interp(np.linspace(900, 1650, 256), SEFD_MHZ, SEFD_JY).mean()
    return scale * sefd / np.sqrt(2.0 * DNU * DT)


def main(args):
    scales = [float(s) for s in args.scales.split()]
    root = Path(args.runs_root)
    field = args.field
    ch = args.channel

    handles, n_min = [], None
    for s in scales:
        h5 = root / f"runthr_n{int(round(s * 1000)):04d}" / 'dataset.h5'
        if not h5.exists():
            print(f"scale={s}: missing {h5}", flush=True); handles.append(None); continue
        f = h5py.File(h5, 'r')
        handles.append((s, f))
        n_min = f[field].shape[0] if n_min is None else min(n_min, f[field].shape[0])
        print(f"scale={s}: {h5}  {field}{f[field].shape}", flush=True)
    present = [h for h in handles if h]
    if not present:
        raise SystemExit("no datasets found")

    if args.tiles:
        tiles = [int(t) for t in args.tiles.split()]
    else:
        tiles = list(np.linspace(0, n_min - 1, args.n_show).astype(int))
    print(f"tiles={tiles}", flush=True)

    ncols = len(present)
    fig, axes = plt.subplots(len(tiles), ncols, figsize=(3.3 * ncols, 3.0 * len(tiles)))
    axes = np.atleast_2d(axes)
    for ri, t in enumerate(tiles):
        ref = present[0][1][field][t, ch]
        vmin, vmax = np.percentile(ref, 1), np.percentile(ref, 99)
        for ci, (s, f) in enumerate(present):
            img = f[field][t, ch]
            ax = axes[ri, ci]
            ax.imshow(img.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"{s}x SEFD\nsigma={sigma_of(s):.3f} Jy", fontsize=9)
        axes[ri, 0].set_ylabel(f"tile {t}", fontsize=9)

    fig.suptitle(f"raw training data ({field} ch{ch}) — same tile, increasing thermal noise",
                 fontsize=11, y=1.0)
    plt.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=115, bbox_inches='tight')
    print(f"-> {out}", flush=True)
    for h in present:
        h[1].close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--scales', required=True)
    ap.add_argument('--field', default='clean', choices=['clean', 'corrupted'])
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--n-show', type=int, default=4, dest='n_show')
    ap.add_argument('--tiles', default='')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
