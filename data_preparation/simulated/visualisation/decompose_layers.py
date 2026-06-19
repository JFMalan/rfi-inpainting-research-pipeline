import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d, gaussian_filter


def smooth_and_residual(amp, mask, bins=8, sigma=2.0):
    # 2D low-pass: hole-fill along freq first, then a 2D Gaussian blur. A 2D kernel keeps
    # DIAGONAL fringe structure in the smooth (recoverable) component, unlike a 1D freq
    # box-filter which smears diagonals into the residual. sigma sets the cutoff scale.
    filled = amp.copy()
    nf = amp.shape[1]
    idx = np.arange(nf)
    for t in range(amp.shape[0]):
        row = filled[t]; keep = mask[t] < 0.5
        if keep.sum() < 4:
            filled[t] = row.mean() if keep.any() else 1.0
            continue
        filled[t] = np.interp(idx, idx[keep], row[keep])
    out = gaussian_filter(filled, sigma=sigma, mode='nearest')
    return out.astype(np.float32), (filled - out).astype(np.float32)


def lag1(x, axis):
    a = x if axis == 1 else x.T
    x0, x1 = a[:, :-1].ravel(), a[:, 1:].ravel()
    x0 = x0 - x0.mean(); x1 = x1 - x1.mean()
    d = np.sqrt((x0 * x0).sum() * (x1 * x1).sum())
    return float((x0 * x1).sum() / d) if d > 0 else float('nan')


def main(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, 'r') as f:
        akey = 'clean' if 'clean' in f else 'data'
        mkey = 'mask' if 'mask' in f else 'flags'
        tot = f[akey].shape[0]
        fmin = float(f.attrs['freq_min_mhz']); fmax = float(f.attrs['freq_max_mhz'])
        if args.patches:
            idx = [int(p) for p in args.patches.split(',')]
        else:
            idx = np.linspace(0, tot - 1, args.n).astype(int).tolist()
        print(f"{akey}: {tot} patches, using {idx}", flush=True)
        ext = [0, f[akey].shape[1], fmin, fmax]
        ac_s, ac_g, ac_p = [], [], []
        for p in idx:
            amp = f[akey][p].astype(np.float32)
            mask = f[mkey][p].astype(np.float32)
            phase = f['phase'][p].astype(np.float32)
            smooth, grain = smooth_and_residual(amp, mask, args.bins, args.sigma)
            cos_p = np.cos(phase)
            valid = mask < 0.5
            s_ac, g_ac, p_ac = lag1(smooth, 1), lag1(grain, 1), lag1(cos_p, 1)
            ac_s.append(s_ac); ac_g.append(g_ac); ac_p.append(p_ac)
            print(f"patch {p:5d}  flag {mask.mean():.2f}  std amp={amp[valid].std():.3f} "
                  f"smooth={smooth[valid].std():.3f} grain={grain[valid].std():.3f}  "
                  f"ac(freq) smooth={s_ac:.2f} grain={g_ac:.2f} phase={p_ac:.2f}", flush=True)
            vlo, vhi = np.percentile(amp[valid], [2, 98])
            gs = max(grain.std(), 1e-6)
            panels = [
                (amp.T, 'observed amplitude', vlo, vhi, 'plasma'),
                (smooth.T, f'SMOOTH amp (recoverable, ac={s_ac:.2f})', vlo, vhi, 'plasma'),
                (grain.T, f'amp GRAIN (noise, ac={g_ac:.2f})', -3*gs, 3*gs, 'coolwarm'),
                (cos_p.T, f'cos(phase) FRINGES (recoverable, ac={p_ac:.2f})', -1, 1, 'twilight'),
            ]
            fig, ax = plt.subplots(1, 4, figsize=(20, 5))
            for a, (im, title, lo, hi, cm) in zip(ax, panels):
                a.imshow(im, aspect='auto', origin='lower', extent=ext, vmin=lo, vmax=hi, cmap=cm)
                a.set_title(title, fontsize=10); a.set_xlabel('time')
            ax[0].set_ylabel('freq MHz')
            fig.suptitle(f'patch {p}  (flag {mask.mean():.2f})', fontsize=11)
            plt.tight_layout(); plt.savefig(out / f'layers_patch{p}.png', dpi=110); plt.close()

    print(f"\nMEAN over {len(idx)} patches  ac(freq):  smooth={np.nanmean(ac_s):.3f}  "
          f"grain={np.nanmean(ac_g):.3f}  phase={np.nanmean(ac_p):.3f}", flush=True)
    print("  -> grain ~0 = irreducible noise; smooth & phase high = recoverable structure", flush=True)
    print(f"saved {len(idx)} figures -> {out}/", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--patches', default=None, help='comma-separated indices; overrides --n')
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--sigma', type=float, default=2.0, help='2D Gaussian low-pass cutoff')
    ap.add_argument('--output', default='vis-layers')
    main(ap.parse_args())
