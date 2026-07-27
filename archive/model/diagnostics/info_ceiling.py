import argparse

import numpy as np
import h5py
from scipy.ndimage import uniform_filter, binary_erosion


def interp_fill(a, h):
    out = a.copy(); nt, nf = a.shape; idx = np.arange(nf)
    for t in range(nt):
        hr = h[t]
        if hr.any() and not hr.all():
            out[t, hr] = np.interp(idx[hr], idx[~hr], a[t, ~hr])
    return out


def main(args):
    f = h5py.File(args.data, 'r')
    n = min(args.n, f['clean'].shape[0])
    clean = f['clean'][:n]
    mask = f['mask'][:n] > 0

    # E1a: context->hole predictability. Regress each hole pixel's value on the
    # local context mean (ring around it). R^2 ~ 0 => in-hole amplitude is not
    # predictable from context (white-noise-like); R^2 high => recoverable structure.
    hole_vals, ctx_vals = [], []
    edge_err_interp, int_err_interp, edge_err_mf, int_err_mf = [], [], [], []
    autocorr_lag1 = []
    for i in range(n):
        c = clean[i]; h = mask[i]
        if h.sum() < 10:
            continue
        # local context estimate = smoothed clean with holes interpolated first
        ctx = uniform_filter(interp_fill(c, h), size=5, mode='nearest')
        hole_vals.append(c[h]); ctx_vals.append(ctx[h])

        # edge vs interior of holes: erode the mask; interior = eroded, edge = ring
        interior = binary_erosion(h, iterations=2)
        edge = h & ~interior
        mu = c[~h].mean()
        ip = interp_fill(c, h)
        if edge.any():
            edge_err_interp.append(np.abs(ip - c)[edge].mean())
            edge_err_mf.append(np.abs(mu - c)[edge].mean())
        if interior.any():
            int_err_interp.append(np.abs(ip - c)[interior].mean())
            int_err_mf.append(np.abs(mu - c)[interior].mean())

        # spatial autocorrelation of the clean amplitude (lag-1 along freq) in known region
        row = c.copy(); row[h] = np.nan
        a = row[:, :-1].ravel(); b = row[:, 1:].ravel()
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() > 100:
            autocorr_lag1.append(np.corrcoef(a[ok], b[ok])[0, 1])

    hv = np.concatenate(hole_vals); cv = np.concatenate(ctx_vals)
    # R^2 of predicting hole value from local context
    ss_res = ((hv - cv) ** 2).sum()
    ss_tot = ((hv - hv.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    print(f"patches: {n}   clean amp std: {clean.std():.4f}")
    print("\nE1  information ceiling (is in-hole amplitude predictable from context?)")
    print(f"  context->hole R^2          : {r2:.4f}   (near 0 => not predictable; near 1 => recoverable)")
    print(f"  lag-1 freq autocorrelation : {np.mean(autocorr_lag1):.4f}   (near 0 => white-noise-like)")
    print(f"  hole EDGE   : interp {np.mean(edge_err_interp):.4f}  mean-fill {np.mean(edge_err_mf):.4f}")
    print(f"  hole INTERIOR: interp {np.mean(int_err_interp):.4f}  mean-fill {np.mean(int_err_mf):.4f}")
    print("  (if interp beats mean-fill at EDGES but ties INTERIOR -> structure is only")
    print("   local-edge, interior is unrecoverable noise -> amplitude not generalizably inpaintable)")

    # E5: metric alignment. Our hole-only MAE vs whole-image PSNR (the paper's metric)
    # using mean-fill as a stand-in prediction (known region perfect, hole=mean).
    print("\nE5  metric alignment (our hole-MAE vs paper-style whole-image PSNR)")
    dr = float(clean.max() - clean.min())
    hole_mae, whole_mae = [], []
    for i in range(n):
        c = clean[i]; h = mask[i]
        if h.sum() < 10:
            continue
        pred = c.copy(); pred[h] = c[~h].mean()   # mean-fill in hole, perfect outside
        hole_mae.append(np.abs(pred - c)[h].mean())
        whole_mae.append(np.abs(pred - c).mean())  # whole image (mostly perfect)
    hm = np.mean(hole_mae); wm = np.mean(whole_mae)
    print(f"  hole-only MAE (our metric)     : {hm:.4f}")
    whole_mse = wm ** 2
    print(f"  whole-image PSNR (paper metric): {20*np.log10(dr/np.sqrt(whole_mse+1e-12)):.1f} dB")
    print("  (a mean-fill 'prediction' already looks great on whole-image PSNR because")
    print("   most of the image is the perfect known region -> their headline PSNR is inflated)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=300)
    main(ap.parse_args())
