import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


def load(path):
    d = fits.getdata(path)
    return np.squeeze(d).astype(np.float32)


def stats(img):
    mean, med, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)
    peak = float(np.nanmax(img))
    return {'rms': float(std), 'peak': peak, 'dr': float(peak / std) if std > 0 else 0.0}


def main(args):
    imgs = {'clean': args.clean, 'flagged': args.flagged, 'meanfill': args.meanfill,
            'classical': args.classical, 'inpainted': args.inpainted}
    imgs = {k: v for k, v in imgs.items() if v}
    data = {k: load(v) for k, v in imgs.items()}

    print(f"{'image':<12}{'off-src RMS':>14}{'peak':>12}{'dyn.range':>12}", flush=True)
    s = {}
    for k, img in data.items():
        s[k] = stats(img)
        print(f"{k:<12}{s[k]['rms']:>14.3e}{s[k]['peak']:>12.4f}{s[k]['dr']:>12.1f}", flush=True)

    metrics = {k: dict(s[k]) for k in s}
    if 'clean' in data:
        ref = data['clean']
        print("\nfidelity vs clean (lower = closer to truth):", flush=True)
        for k in data:
            if k == 'clean':
                continue
            rmse = float(np.sqrt(np.nanmean((data[k] - ref) ** 2)))
            metrics[k]['rmse_vs_clean'] = rmse
            print(f"  {k:<10} image RMSE {rmse:.3e}", flush=True)
        if {'flagged', 'inpainted'} <= set(data):
            better = "INPAINTED closer to clean" if \
                np.nanmean((data['inpainted'] - ref) ** 2) < np.nanmean((data['flagged'] - ref) ** 2) \
                else "flagged closer to clean"
            print(f"  verdict: {better}", flush=True)

    order = [k for k in ['flagged', 'meanfill', 'classical', 'inpainted', 'clean'] if k in data]
    fig, ax = plt.subplots(1, len(order), figsize=(6 * len(order), 5.5))
    if len(order) == 1:
        ax = [ax]
    ref = data.get('clean')
    vlo, vhi = np.nanpercentile(ref if ref is not None else data[order[0]], [5, 99.5])
    for a, k in zip(ax, order):
        a.imshow(data[k], origin='lower', cmap='inferno', vmin=vlo, vmax=vhi)
        a.set_title(f"{k}\nRMS {s[k]['rms']:.2e}  DR {s[k]['dr']:.0f}", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"\nsaved -> {args.out}", flush=True)
    if args.metrics_out:
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, default=float))
        print(f"metrics -> {args.metrics_out}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', default='')
    ap.add_argument('--flagged', default='')
    ap.add_argument('--meanfill', default='')
    ap.add_argument('--classical', default='')
    ap.add_argument('--inpainted', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--metrics-out', default=None, dest='metrics_out')
    main(ap.parse_args())
