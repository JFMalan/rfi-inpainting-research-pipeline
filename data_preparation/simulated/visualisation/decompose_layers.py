import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d


def smooth_and_residual(amp, mask, bins=8):
    out = amp.copy()
    nf = amp.shape[1]
    idx = np.arange(nf)
    for t in range(amp.shape[0]):
        row = amp[t]; keep = mask[t] < 0.5
        if keep.sum() < 4:
            out[t] = row.mean() if keep.any() else 1.0
            continue
        out[t] = uniform_filter1d(np.interp(idx, idx[keep], row[keep]), size=bins, mode='nearest')
    return out.astype(np.float32), (amp - out).astype(np.float32)


def lag1(x, axis):
    a = x if axis == 1 else x.T
    x0, x1 = a[:, :-1].ravel(), a[:, 1:].ravel()
    x0 = x0 - x0.mean(); x1 = x1 - x1.mean()
    d = np.sqrt((x0 * x0).sum() * (x1 * x1).sum())
    return float((x0 * x1).sum() / d) if d > 0 else float('nan')


def main(args):
    with h5py.File(args.data, 'r') as f:
        akey = 'clean' if 'clean' in f else 'data'
        mkey = 'mask' if 'mask' in f else 'flags'
        amp = f[akey][args.patch].astype(np.float32)
        mask = f[mkey][args.patch].astype(np.float32)
        phase = f['phase'][args.patch].astype(np.float32)
        fmin = float(f.attrs['freq_min_mhz']); fmax = float(f.attrs['freq_max_mhz'])

    smooth, grain = smooth_and_residual(amp, mask, args.bins)
    cos_p, sin_p = np.cos(phase), np.sin(phase)
    valid = mask < 0.5

    print(f"patch {args.patch}  flag frac {mask.mean():.3f}", flush=True)
    print(f"layer stds (unflagged):  amp={amp[valid].std():.3f}  smooth={smooth[valid].std():.3f}  "
          f"grain={grain[valid].std():.3f}", flush=True)
    print(f"lag-1 autocorr along-freq:  smooth={lag1(smooth,1):.3f}  grain={lag1(grain,1):.3f}  "
          f"cos(phase)={lag1(cos_p,1):.3f}", flush=True)
    print(f"  (high autocorr = recoverable structure; ~0 = white noise)", flush=True)

    ext = [0, amp.shape[0], fmin, fmax]
    vlo, vhi = np.percentile(amp[valid], [2, 98])
    panels = [
        (amp.T, 'observed amplitude', vlo, vhi, 'plasma'),
        (smooth.T, f'SMOOTH amp (recoverable, ac={lag1(smooth,1):.2f})', vlo, vhi, 'plasma'),
        (grain.T, f'amp GRAIN (noise, ac={lag1(grain,1):.2f})', -3*grain.std(), 3*grain.std(), 'coolwarm'),
        (cos_p.T, f'cos(phase) FRINGES (recoverable, ac={lag1(cos_p,1):.2f})', -1, 1, 'twilight'),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, (img, title, lo, hi, cm) in zip(ax, panels):
        a.imshow(img, aspect='auto', origin='lower', extent=ext, vmin=lo, vmax=hi, cmap=cm)
        a.set_title(title, fontsize=10); a.set_xlabel('time')
    ax[0].set_ylabel('freq MHz')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out / f'layers_patch{args.patch}.png', dpi=120); plt.close()
    print(f"saved -> {out}/layers_patch{args.patch}.png", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--patch', type=int, default=0)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--output', default='vis-layers')
    main(ap.parse_args())
