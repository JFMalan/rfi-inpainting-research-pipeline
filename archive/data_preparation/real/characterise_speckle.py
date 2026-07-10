import argparse
import time
import glob
import h5py
import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def decompose(img, valid, smooth=8):
    # split a waterfall into a smooth (recoverable) component and a residual (speckle).
    # smooth = box average over freq of the unflagged signal; residual = img - smooth.
    filled = img.copy()
    n_time, n_freq = img.shape
    idx = np.arange(n_freq)
    for t in range(n_time):
        row = filled[t]; v = valid[t]
        if v.sum() < 4:
            filled[t] = row.mean() if v.any() else 1.0
            continue
        filled[t] = np.interp(idx, idx[v], row[v])
    sm = uniform_filter1d(filled, size=smooth, axis=1, mode='nearest')
    res = filled - sm
    return sm, res, filled


def lag1(res, valid, axis):
    a = res if axis == 1 else res.T
    m = valid if axis == 1 else valid.T
    x0 = a[:, :-1][m[:, :-1] & m[:, 1:]]
    x1 = a[:, 1:][m[:, :-1] & m[:, 1:]]
    if x0.size < 50:
        return np.nan
    x0 = x0 - x0.mean(); x1 = x1 - x1.mean()
    d = np.sqrt((x0 * x0).sum() * (x1 * x1).sum())
    return (x0 * x1).sum() / d if d > 0 else np.nan


def corr_at_lag(res, valid, lag):
    x0 = res[:, :-lag][valid[:, :-lag] & valid[:, lag:]]
    x1 = res[:, lag:][valid[:, :-lag] & valid[:, lag:]]
    if x0.size < 50:
        return np.nan
    x0 = x0 - x0.mean(); x1 = x1 - x1.mean()
    d = np.sqrt((x0 * x0).sum() * (x1 * x1).sum())
    return (x0 * x1).sum() / d if d > 0 else np.nan


def main(args):
    files = sorted(f for p in args.files for f in glob.glob(p))
    log(f"files: {files}")
    key, fkey = ('data', 'flags')
    with h5py.File(files[0], 'r') as f:
        if 'data' not in f:
            key, fkey = ('clean', 'mask')
    log(f"using key={key} flags={fkey}")

    imgs, flags = [], []
    with h5py.File(files[0], 'r') as f:
        tot = f[key].shape[0]
        idx = np.linspace(0, tot - 1, min(args.n, tot)).astype(int)
        imgs = f[key][idx].astype(np.float32)
        flags = f[fkey][idx].astype(np.float32)
    valid = flags < 0.5
    log(f"loaded {imgs.shape}, flag frac {1 - valid.mean():.3f}")

    sm_std, res_std, total_std = [], [], []
    ac_f, ac_t = [], []
    lag_curve = np.zeros(args.maxlag)
    lag_cnt = np.zeros(args.maxlag)
    res_frac = []

    for im, v in zip(imgs, valid):
        if v.sum() < 1000:
            continue
        sm, res, _ = decompose(im, v, smooth=args.smooth)
        sm_std.append(sm[v].std())
        res_std.append(res[v].std())
        total_std.append(im[v].std())
        res_frac.append(res[v].std() / (im[v].std() + 1e-8))
        ac_f.append(lag1(res, v, 1))
        ac_t.append(lag1(res, v, 0))
        for L in range(1, args.maxlag + 1):
            c = corr_at_lag(res, v, L)
            if np.isfinite(c):
                lag_curve[L - 1] += c; lag_cnt[L - 1] += 1

    lag_curve = lag_curve / np.maximum(lag_cnt, 1)
    print("\n=== SPECKLE CHARACTERISATION (real) ===", flush=True)
    print(f"smooth-bins for decompose: {args.smooth}", flush=True)
    print(f"total amp std (unflagged)   : {np.nanmean(total_std):.4f}", flush=True)
    print(f"smooth component std        : {np.nanmean(sm_std):.4f}", flush=True)
    print(f"residual (speckle) std      : {np.nanmean(res_std):.4f}", flush=True)
    print(f"speckle / total std ratio   : {np.nanmean(res_frac):.4f}", flush=True)
    print(f"residual lag-1 autocorr freq: {np.nanmean(ac_f):.4f}", flush=True)
    print(f"residual lag-1 autocorr time: {np.nanmean(ac_t):.4f}", flush=True)
    print("\nresidual autocorr vs freq-lag (correlation length):", flush=True)
    for L in range(args.maxlag):
        print(f"  lag {L+1:2d}: {lag_curve[L]:.4f}", flush=True)
    # correlation length where autocorr drops below 1/e
    below = np.where(lag_curve < np.exp(-1))[0]
    clen = (below[0] + 1) if below.size else args.maxlag
    print(f"\nestimated freq correlation length (1/e): ~{clen} channels", flush=True)
    print("\n--> to mimic: residual ~ Gaussian noise, std ="
          f" {np.nanmean(res_std):.4f}, smoothed with kernel ~{clen} channels along freq", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+',
                    default=['/scratch3/users/jfmalan/rfi/real/variants/v1_upsample512.h5'])
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--smooth', type=int, default=8)
    ap.add_argument('--maxlag', type=int, default=12)
    main(ap.parse_args())
