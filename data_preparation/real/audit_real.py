import argparse

import numpy as np
import h5py
from scipy.ndimage import uniform_filter, binary_erosion


def fake_mask(real_flags, frac_range=(0.05, 0.25), max_tries=200):
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


def section(title):
    print(f"\n{'='*64}\n{title}\n{'='*64}", flush=True)


def main(args):
    f = h5py.File(args.data, 'r')
    n = min(args.n, f['data'].shape[0])
    amp = f['data'][:n].astype(np.float32)
    flags = f['flags'][:n].astype(np.float32)
    phase = f['phase'][:n].astype(np.float32) if 'phase' in f else None
    div = f['dn_divisor'][:n].astype(np.float32) if 'dn_divisor' in f else None
    f.close()
    unflag = flags < 0.5

    # ---- H1: amplitude scale + outliers ----
    section("H1  amplitude scale + outliers (DN'd 'data' field)")
    a_all = amp.ravel()
    a_un = amp[unflag]
    print(f"  ALL pixels   : mean {a_all.mean():.3f}  std {a_all.std():.3f}  "
          f"p50 {np.percentile(a_all,50):.3f}  p99 {np.percentile(a_all,99):.3f}  "
          f"p99.9 {np.percentile(a_all,99.9):.3f}  max {a_all.max():.1f}")
    print(f"  UNFLAGGED    : mean {a_un.mean():.3f}  std {a_un.std():.3f}  "
          f"p50 {np.percentile(a_un,50):.3f}  p99 {np.percentile(a_un,99):.3f}  "
          f"p99.9 {np.percentile(a_un,99.9):.3f}  max {a_un.max():.1f}")
    for thr in (2, 5, 10, 50):
        fr = (a_un > thr).mean()
        print(f"  unflagged pixels > {thr:>3}: {fr*100:.3f}%")
    print("  -> if a tiny % of unflagged pixels are huge, those are residual RFI / band-edge")
    print("     spikes the flagging missed; they blow up std + mean-fill + R^2.")

    # ---- H2: are outliers spatially structured (band edges) or scattered (RFI)? ----
    section("H2  where are the extreme unflagged pixels?")
    big = (amp > 5) & unflag
    if big.any():
        per_chan = big.reshape(-1, big.shape[-1]).mean(axis=0)
        top = np.argsort(per_chan)[-8:][::-1]
        print(f"  fraction of big-pixel rows by freq-channel (top 8): {[(int(c), round(float(per_chan[c]),4)) for c in top]}")
        print("  (concentrated in few channels -> band-edge/persistent residual; spread -> scattered RFI)")
    else:
        print("  no unflagged pixels > 5")

    # ---- H3: recoverability, RAW vs OUTLIER-CLIPPED ----
    section("H3  recoverability (mean-fill vs interp; raw and clipped)")
    rng = np.random.default_rng(0)
    for clip in (None, 5.0):
        hole_vals, ctx_vals, autoc = [], [], []
        mf_e, ip_e = [], []
        for i in range(n):
            c = amp[i].copy(); rf = flags[i]
            if clip is not None:
                c = np.clip(c, None, clip)
            np.random.seed(int(rng.integers(0, 1 << 31)))
            h = fake_mask(rf) > 0.5
            if h.sum() < 10 or (~h).sum() < 100:
                continue
            ctx = uniform_filter(interp_fill(c, h), size=5, mode='nearest')
            hole_vals.append(c[h]); ctx_vals.append(ctx[h])
            mu = c[~h].mean(); ip = interp_fill(c, h)
            mf_e.append(np.abs(mu - c)[h].mean()); ip_e.append(np.abs(ip - c)[h].mean())
            row = c.copy(); row[h] = np.nan
            a = row[:, :-1].ravel(); b = row[:, 1:].ravel(); ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() > 100:
                autoc.append(np.corrcoef(a[ok], b[ok])[0, 1])
        hv = np.concatenate(hole_vals); cv = np.concatenate(ctx_vals)
        r2 = 1 - ((hv - cv) ** 2).sum() / ((hv - hv.mean()) ** 2).sum()
        tag = "RAW" if clip is None else f"CLIP<{clip}"
        print(f"  [{tag}] R^2 {r2:.4f}  autocorr {np.mean(autoc):.4f}  "
              f"mean-fill MAE {np.mean(mf_e):.4f}  interp MAE {np.mean(ip_e):.4f}  "
              f"interp/mf {np.mean(ip_e)/max(np.mean(mf_e),1e-9):.3f}")
    print("  -> interp << mean-fill => structure recoverable. clipping should make R^2 sane")
    print("     if outliers were the only problem.")

    # ---- H4: reconcile with the comparison's 3-channel metric ----
    section("H4  per-channel scale (why comparison mean-fill ~0.16 vs raw amp ~5)")
    if phase is not None:
        cosp, sinp = np.cos(phase), np.sin(phase)
        print(f"  amp   : mean {amp.mean():.3f}  std {amp.std():.3f}")
        print(f"  cos   : mean {cosp.mean():.3f}  std {cosp.std():.3f}  (bounded [-1,1])")
        print(f"  sin   : mean {sinp.mean():.3f}  std {sinp.std():.3f}  (bounded [-1,1])")
        print("  -> the comparison's mae() uses CHANNEL 0 only (amp), but if obs amp here is")
        print("     huge-std, a 0.16 fake-MAE in the comparison means the model DID fill amp well")
        print("     OR the metric scale differs. cross-check channel-0 mean-fill below:")
        mf_ch0 = []
        rng2 = np.random.default_rng(1)
        for i in range(min(n, 64)):
            rf = flags[i]; np.random.seed(int(rng2.integers(0, 1 << 31)))
            h = fake_mask(rf) > 0.5
            if h.sum() < 10 or (~h).sum() < 100:
                continue
            mu = amp[i][~h].mean()
            mf_ch0.append(np.abs(mu - amp[i])[h].mean())
        print(f"  channel-0 (amp) mean-fill MAE: {np.mean(mf_ch0):.4f}")

    # ---- H5: DN divisor sanity ----
    section("H5  divisive-norm divisor (did DN do its job?)")
    if div is not None:
        print(f"  divisor: mean {div.mean():.3f}  std {div.std():.3f}  "
              f"min {div.min():.4f}  max {div.max():.1f}")
        print(f"  amp/divisor would be raw Jy; amp here is ALREADY normed (amp=raw/divisor).")
        print(f"  if amp still has 108-std outliers AFTER dividing, DN's smooth divisor did not")
        print(f"  track the spikes (expected: spikes are narrow, the smoother averages them out).")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/scratch3/users/jfmalan/rfi/real/variants/v1_upsample512.h5')
    ap.add_argument('--n', type=int, default=200)
    main(ap.parse_args())
