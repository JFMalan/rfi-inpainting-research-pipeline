import argparse

import numpy as np
import h5py
from scipy.ndimage import uniform_filter, binary_erosion


def fake_mask(real_flags, frac_range=(0.05, 0.25), max_tries=200):
    # identical to model/data.py fake_mask (inlined: that module imports torch,
    # unavailable in the data-prep container). Holes over UNFLAGGED pixels only.
    n_time, n_freq = real_flags.shape
    fm = np.zeros((n_time, n_freq), dtype=np.float32)
    target = np.random.uniform(*frac_range)
    unflagged = real_flags < 0.5
    target = min(target, max(unflagged.mean() * 0.8, 1e-3))
    tries = 0
    while (fm * unflagged).mean() < target and tries < max_tries:
        tries += 1
        if np.random.rand() < 0.6:
            w = np.random.randint(1, 8); f0 = np.random.randint(0, n_freq - w)
            fm[:, f0:f0 + w] = 1.0
        else:
            w = np.random.randint(1, 8); t0 = np.random.randint(0, n_time - w)
            fm[t0:t0 + w, :] = 1.0
    return fm * unflagged


def interp_fill(a, h):
    out = a.copy(); nt, nf = a.shape; idx = np.arange(nf)
    for t in range(nt):
        hr = h[t]
        if hr.any() and not hr.all():
            out[t, hr] = np.interp(idx[hr], idx[~hr], a[t, ~hr])
    return out


def recoverability(sig, holes, label):
    hole_vals, ctx_vals, autoc = [], [], []
    edge_i, int_i, edge_m, int_m = [], [], [], []
    for c, h in zip(sig, holes):
        if h.sum() < 10 or (~h).sum() < 100:
            continue
        ctx = uniform_filter(interp_fill(c, h), size=5, mode='nearest')
        hole_vals.append(c[h]); ctx_vals.append(ctx[h])
        interior = binary_erosion(h, iterations=2); edge = h & ~interior
        mu = c[~h].mean(); ip = interp_fill(c, h)
        if edge.any():
            edge_i.append(np.abs(ip - c)[edge].mean()); edge_m.append(np.abs(mu - c)[edge].mean())
        if interior.any():
            int_i.append(np.abs(ip - c)[interior].mean()); int_m.append(np.abs(mu - c)[interior].mean())
        row = c.copy(); row[h] = np.nan
        a = row[:, :-1].ravel(); b = row[:, 1:].ravel()
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() > 100:
            autoc.append(np.corrcoef(a[ok], b[ok])[0, 1])
    hv = np.concatenate(hole_vals); cv = np.concatenate(ctx_vals)
    r2 = 1 - ((hv - cv) ** 2).sum() / ((hv - hv.mean()) ** 2).sum()
    print(f"\n[{label}]  signal std {np.concatenate([s.ravel() for s in sig]).std():.4f}")
    print(f"  context->hole R^2          : {r2:.4f}   (~0 unrecoverable, ~1 recoverable)")
    print(f"  lag-1 freq autocorrelation : {np.mean(autoc):.4f}   (~0 white-noise-like)")
    print(f"  hole EDGE    interp {np.mean(edge_i):.4f}  mean-fill {np.mean(edge_m):.4f}")
    print(f"  hole INTERIOR interp {np.mean(int_i):.4f}  mean-fill {np.mean(int_m):.4f}")
    return r2


def main(args):
    f = h5py.File(args.data, 'r')
    n = min(args.n, f['data'].shape[0])
    data = f['data'][:n].astype(np.float32)
    real_flags = f['flags'][:n].astype(np.float32)
    f.close()

    rng = np.random.default_rng(0)
    fakes = []
    for rf in real_flags:
        np.random.seed(int(rng.integers(0, 1 << 31)))
        fakes.append(fake_mask(rf) > 0.5)
    fakes = np.array(fakes)

    print(f"real baselines: {n}")
    print("Q: do the fake-mask holes (= what the comparison trained/scored on) contain")
    print("   recoverable structure? R^2~0 there => the 'ties mean-fill' result is REAL,")
    print("   not a training/metric bug; R^2 high => we have a model/metric problem.")

    # the thing the comparison actually measured: fake holes over the observed amplitude
    recoverability(list(data), list(fakes), "fake-mask holes over observed amplitude (what we scored)")

    # control: are the REAL-flagged regions (what we ultimately want to fill) different?
    # we can't measure recoverability there (no ground truth), but report how much
    # structure sits in unflagged vs the data overall, to flag the test/target mismatch.
    unflag_std = np.array([d[rf < 0.5].std() for d, rf in zip(data, real_flags) if (rf < 0.5).sum() > 100])
    print(f"\nunflagged-region amplitude std (per-baseline mean): {unflag_std.mean():.4f}")
    print("  (if this is ~the mean-fill MAE scale, the unflagged data is near-featureless ->")
    print("   the fake-mask test is measuring 'predict faint noise', the SNR ceiling is real)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/scratch3/users/jfmalan/rfi/real/variants/v1_upsample512.h5')
    ap.add_argument('--n', type=int, default=200)
    main(ap.parse_args())
