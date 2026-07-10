import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest_sample(run_dir):
    fs = glob.glob(str(run_dir / 'samples' / 'sample_e*.npz'))
    if not fs:
        return None, None
    ep = lambda p: int(re.search(r'sample_e(\d+)', p).group(1))
    f = max(fs, key=ep)
    return f, ep(f)


def row_metrics(run_dir):
    p = run_dir / 'log.json'
    if not p.exists():
        return None
    ev = [e for e in json.loads(p.read_text()) if 'amp_mae' in e]
    if not ev:
        return None
    b = min(ev, key=lambda e: e['amp_mae'])
    return b['amp_mae'], b.get('amp_mf', float('nan')), b.get('beats_mf', False)


def green(mask2d):
    out = np.zeros((*mask2d.T.shape, 4), np.float32)
    out[mask2d.T > 0] = [0.0, 1.0, 0.2, 0.7]
    return out


def main(args):
    scales = [float(s) for s in args.scales.split()]
    root = Path(args.runs_root)
    rows = []
    for s in scales:
        rd = root / f"phase1_thr_n{int(round(s * 1000)):04d}"
        f, ep = latest_sample(rd)
        if f is None:
            print(f"scale={s}: no samples yet in {rd}", flush=True)
            continue
        d = np.load(f)
        rows.append((s, ep, d, row_metrics(rd)))
        print(f"scale={s}: {f} (epoch {ep})", flush=True)
    if not rows:
        raise SystemExit("no sample npz found for any scale yet")

    nt = min(args.n_show, rows[0][2]['clean'].shape[0])
    ncols = 3 * nt
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.1 * ncols, 3.0 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (s, ep, d, m) in enumerate(rows):
        clean, corr, mask, pred = d['clean'], d['corrupted'], d['mask'], d['pred']
        for t in range(nt):
            truth = clean[t, 0]
            obs = corr[t, 0]
            mk = mask[t, 0] > 0.5
            fill = np.where(mk, pred[t, 0], truth)
            vmin, vmax = np.percentile(truth, 1), np.percentile(truth, 99)
            base = 3 * t
            for j, (img, tag) in enumerate([(obs, 'observed'), (truth, 'truth'), (fill, 'model fill')]):
                ax = axes[r, base + j]
                ax.imshow(img.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
                if j == 0:
                    ax.imshow(green(mk), aspect='auto', origin='lower')
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"tile {t} {tag}", fontsize=9)
        lab = f"{s}x SEFD\nep {ep}"
        if m:
            lab += f"\nMAE {m[0]:.4f}\nmf {m[1]:.4f}\n{'BEATS mf' if m[2] else 'no'}"
        axes[r, 0].set_ylabel(lab, fontsize=8, rotation=0, ha='right', va='center')

    plt.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=115, bbox_inches='tight')
    print(f"-> {out}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--scales', required=True)
    ap.add_argument('--n-show', type=int, default=2, dest='n_show')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
