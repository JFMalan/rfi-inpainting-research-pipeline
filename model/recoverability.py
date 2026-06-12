import argparse

import numpy as np
import h5py
from scipy.ndimage import uniform_filter


def mae(pred, clean, region):
    return float(np.abs(pred - clean)[region].mean())


def biharmonic_fill(img, hole):
    # cheap iterative Laplace/biharmonic-style inpaint: relax known boundary inward
    out = img.copy()
    out[hole] = img[~hole].mean()
    for _ in range(200):
        lap = uniform_filter(out, size=3, mode='nearest')
        out[hole] = lap[hole]
    return out


def interp_fill(img, hole):
    # per-row linear interpolation across the hole along frequency
    out = img.copy()
    n_t, n_f = img.shape
    idx = np.arange(n_f)
    for t in range(n_t):
        h = hole[t]
        if h.all() or not h.any():
            continue
        out[t, h] = np.interp(idx[h], idx[~h], img[t, ~h])
    return out


def main(args):
    f = h5py.File(args.data, 'r')
    n = min(args.n, f['clean'].shape[0])
    clean = f['clean'][:n]
    mask = f['mask'][:n]

    res = {'mean_fill': [], 'biharmonic': [], 'interp': [], 'smooth8': [], 'smooth4': [], 'smooth2': []}
    for i in range(n):
        c = clean[i]; h = mask[i] > 0
        if h.sum() < 10:
            continue
        mu = c[~h].mean()
        res['mean_fill'].append(mae(np.full_like(c, mu), c, h))
        res['biharmonic'].append(mae(biharmonic_fill(c, h), c, h))
        res['interp'].append(mae(interp_fill(c, h), c, h))
        for k, s in [('smooth8', 8), ('smooth4', 4), ('smooth2', 2)]:
            res[k].append(mae(uniform_filter(c, size=s, mode='nearest'), c, h))

    print(f"patches: {len(res['mean_fill'])}   clean amp std: {clean.std():.4f}")
    print("\nmask-region MAE by method (lower = more recoverable structure):")
    mf = np.mean(res['mean_fill'])
    for k in ['mean_fill', 'interp', 'biharmonic', 'smooth8', 'smooth4', 'smooth2']:
        v = np.mean(res[k])
        tag = "  <- baseline" if k == 'mean_fill' else (f"  beats mean-fill by {mf-v:+.4f}" if v < mf else "")
        print(f"  {k:12s}: {v:.4f}{tag}")
    print()
    best = min(np.mean(res[k]) for k in ['interp', 'biharmonic'])
    if best < mf - 0.005:
        print(f"=> STRUCTURE IS RECOVERABLE: classical methods beat mean-fill by {mf-best:.4f}.")
        print("   The amplitude is NOT pure noise; a model SHOULD be able to beat mean-fill.")
    else:
        print("=> classical inpainters do NOT beat mean-fill -> masked amplitude is")
        print("   genuinely noise-like at this scale. Then no model can do much better.")
    print("\nNOTE: smoothN rows show recoverable structure at scale N. If smooth2 << smooth8,")
    print("real structure is fine-scale and the 8-px noise-floor estimate was too pessimistic.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=200)
    main(ap.parse_args())
