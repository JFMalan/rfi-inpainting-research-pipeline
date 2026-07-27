import argparse
import sys
import time
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'data_preparation' / 'real'))
from rfi_bands import LBAND_PERSISTENT_MHZ

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def correlated_speckle(shape, std, corr_len, rng):
    # Gaussian noise smoothed along freq to a target correlation length, then
    # rescaled to the target std. corr_len in channels; sigma chosen so the smoothed
    # noise lag-1 autocorr matches real (~0.25 at corr_len~2).
    n_time, n_freq = shape
    w = rng.standard_normal((n_time, n_freq)).astype(np.float32)
    if corr_len > 1:
        sigma = corr_len / 2.0
        w = gaussian_filter1d(w, sigma=sigma, axis=1, mode='nearest')
    w = w / (w.std() + 1e-8) * std
    return w.astype(np.float32)


def rescale_amp(clean, target_mean, target_std, src_mean=None, src_std=None):
    # affine map of the divnormed amplitude so its mean/std match real, preserving the
    # smooth structure. clamped at 0 (amplitude is non-negative).
    sm = clean.mean() if src_mean is None else src_mean
    ss = clean.std() if src_std is None else src_std
    out = (clean - sm) / (ss + 1e-8) * target_std + target_mean
    return np.clip(out, 0.0, None).astype(np.float32)


def persistent_bands(n_freq, fmin, fmax):
    f = np.linspace(fmin, fmax, n_freq)
    out = []
    for lo, hi in LBAND_PERSISTENT_MHZ:
        idx = np.where((f >= lo) & (f <= hi))[0]
        if idx.size:
            out.append((int(idx[0]), int(idx[-1] + 1)))
    return out


def make_mask(n_time, n_freq, bands, target_frac, band_fill, wide_lo, wide_hi, rng):
    # real-like flag morphology: persistent bands almost fully wiped, plus wide
    # contiguous freq bands. No 1-2px stripes (those are trivially interp-solved).
    m = np.zeros((n_time, n_freq), dtype=np.float32)
    for f0, f1 in bands:
        if rng.random() < band_fill:
            m[:, f0:f1] = 1.0
        else:
            t = 0
            while t < n_time:
                on = rng.integers(int(n_time * 0.3), n_time + 1)
                m[t:min(t + on, n_time), f0:f1] = 1.0
                t += on + rng.integers(1, int(n_time * 0.2) + 1)
    guard = 0
    while m.mean() < target_frac and guard < 300:
        guard += 1
        w = int(rng.integers(wide_lo, wide_hi + 1))
        f0 = int(rng.integers(0, max(1, n_freq - w)))
        m[:, f0:f0 + w] = 1.0
    return m


def main(args):
    rng = np.random.default_rng(args.seed)
    fin = h5py.File(args.input, 'r')
    src_key = 'clean' if 'clean' in fin else 'data'
    n = fin[src_key].shape[0]
    nt = int(fin.attrs['n_time']); nf = int(fin.attrs['n_freq'])
    fmin = float(fin.attrs['freq_min_mhz']); fmax = float(fin.attrs['freq_max_mhz'])
    bands = persistent_bands(nf, fmin, fmax)
    log(f"input {args.input}: {n} baselines {nt}x{nf}, {len(bands)} persistent bands")
    log(f"speckle std={args.speckle_std} corr_len={args.corr_len}  "
        f"amp target mean={args.amp_mean} std={args.amp_std}  "
        f"target flag frac={args.target_frac} band_fill={args.band_fill}")

    keep_keys = [k for k in fin if k not in ('clean', 'data', 'corrupted', 'mask', 'flags')]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)

    chunk = 128
    sstats, mstats = [], []
    with h5py.File(out, 'w') as f:
        shp = fin[src_key].shape
        clean_ds = f.create_dataset('clean', shape=shp, dtype=np.float32)
        smooth_ds = f.create_dataset('clean_smooth', shape=shp, dtype=np.float32)
        corr_ds = f.create_dataset('corrupted', shape=shp, dtype=np.float32)
        mask_ds = f.create_dataset('mask', shape=shp, dtype=np.float32)
        for k in keep_keys:
            src = fin[k]
            dst = f.create_dataset(k, shape=src.shape, dtype=src.dtype)
            for s in range(0, src.shape[0], chunk):
                e = min(s + chunk, src.shape[0])
                dst[s:e] = src[s:e]
        for k, v in fin.attrs.items():
            f.attrs[k] = v
        f.attrs['realify_seed'] = args.seed
        f.attrs['speckle_std'] = args.speckle_std
        f.attrs['speckle_corr_len'] = args.corr_len
        f.attrs['amp_target_mean'] = args.amp_mean
        f.attrs['amp_target_std'] = args.amp_std

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            block = fin[src_key][s:e].astype(np.float32)
            for j, patch in enumerate(block):
                smooth = rescale_amp(patch, args.amp_mean, args.amp_std)
                scaled = smooth
                if args.speckle_std > 0:
                    scaled = np.clip(smooth + correlated_speckle(smooth.shape, args.speckle_std,
                                                                 args.corr_len, rng), 0.0, None)
                clean_ds[s + j] = scaled
                smooth_ds[s + j] = smooth
                m = make_mask(nt, nf, bands, args.target_frac, args.band_fill,
                              args.wide_lo, args.wide_hi, rng)
                # corrupted: clean + bright RFI in the mask (model never sees it; hidden)
                rfi = np.zeros_like(scaled)
                peak = rng.uniform(5, 50) * args.amp_std
                rfi[m > 0] = peak * rng.uniform(0.4, 1.0, size=int((m > 0).sum()))
                corr_ds[s + j] = scaled + rfi
                mask_ds[s + j] = m
                sstats.append(scaled.std()); mstats.append(float(m.mean()))
            log(f"  {e}/{n}  clean std {np.mean(sstats):.3f}  flag frac {np.mean(mstats):.3f}")

    fin.close()
    log(f"saved -> {out}  (clean std {np.mean(sstats):.3f}, flag frac {np.mean(mstats):.3f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='clean_baselines.h5 or dataset.h5')
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--amp-mean', type=float, default=1.0, dest='amp_mean')
    ap.add_argument('--amp-std', type=float, default=0.21, dest='amp_std')
    ap.add_argument('--speckle-std', type=float, default=0.0, dest='speckle_std')
    ap.add_argument('--corr-len', type=float, default=2.0, dest='corr_len')
    ap.add_argument('--target-frac', type=float, default=0.48, dest='target_frac')
    ap.add_argument('--band-fill', type=float, default=0.9, dest='band_fill')
    ap.add_argument('--wide-lo', type=int, default=8, dest='wide_lo')
    ap.add_argument('--wide-hi', type=int, default=40, dest='wide_hi')
    main(ap.parse_args())
