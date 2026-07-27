import argparse
import sys
import time
from pathlib import Path

import numpy as np
from casacore.tables import table
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'data_preparation' / 'real'))
from rfi_bands import LBAND_PERSISTENT_MHZ


def log(msg):
    print(msg, flush=True)


def read_cube(ms_path, column, field, fmin, fmax, n_chan_cap):
    t0 = time.time()
    ms = table(ms_path, readonly=True)
    if field is not None:
        ms = ms.query(f"FIELD_ID == {field}")
    cols = ms.colnames()
    if column not in cols:
        raise RuntimeError(f"column {column} not in MS ({cols})")

    sw = table(ms_path + '/SPECTRAL_WINDOW')
    freqs_full = sw.getcol('CHAN_FREQ')[0] / 1e6
    sw.close()
    chan_idx = np.where((freqs_full >= fmin) & (freqs_full <= fmax))[0]
    if len(chan_idx) == 0:
        raise RuntimeError(f"no channels in [{fmin},{fmax}] MHz")
    chan_lo, chan_hi = int(chan_idx[0]), int(chan_idx[-1]) + 1
    if chan_hi - chan_lo > n_chan_cap:
        mid = (chan_lo + chan_hi) // 2
        chan_lo, chan_hi = mid - n_chan_cap // 2, mid + n_chan_cap // 2
    freqs = freqs_full[chan_lo:chan_hi]
    n_chan = chan_hi - chan_lo
    log(f"probe window: {freqs[0]:.1f}-{freqs[-1]:.1f} MHz ({n_chan} channels, idx {chan_lo}:{chan_hi})")

    times = ms.getcol('TIME')
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')
    uvw = ms.getcol('UVW')
    n_row = ms.nrows()
    npol = ms.getcell(column, 0).shape[-1]

    V = np.empty((n_row, n_chan), dtype=np.complex64)
    F = np.empty((n_row, n_chan), dtype=bool)
    chunk = 50000
    log(f"reading {column} window in chunks ({n_row} rows)...")
    for s in range(0, n_row, chunk):
        e = min(s + chunk, n_row)
        d = ms.getcolslice(column, [chan_lo, 0], [chan_hi - 1, npol - 1], [], s, e - s)
        f = ms.getcolslice('FLAG', [chan_lo, 0], [chan_hi - 1, npol - 1], [], s, e - s)
        V[s:e] = d.mean(axis=2).astype(np.complex64)
        F[s:e] = f.any(axis=2)
        if s == 0 or (s // chunk) % 10 == 0:
            log(f"  rows {e}/{n_row}")
    ms.close()

    n_time = len(np.unique(times))
    n_bl = n_row // n_time
    keep = n_time * n_bl
    V = V[:keep].reshape(n_time, n_bl, n_chan)
    F = F[:keep].reshape(n_time, n_bl, n_chan)
    uvw = uvw[:keep].reshape(n_time, n_bl, 3)
    ant1_bl = ant1[:n_bl]; ant2_bl = ant2[:n_bl]
    cross = np.where(ant1_bl != ant2_bl)[0]
    log(f"cube {V.shape}  n_time={n_time} n_bl={n_bl} cross_bl={len(cross)}  read {time.time()-t0:.0f}s")
    return V, F, uvw, freqs, cross


def r2(num, den):
    return float(1.0 - num / den) if den > 0 else float('nan')


def main(args):
    rng = np.random.default_rng(0)
    V, F, uvw, freqs, cross = read_cube(args.ms, args.column, args.field,
                                        args.freq_min, args.freq_max, args.n_chan)
    n_time, n_bl, n_chan = V.shape

    good_ts = F.mean(axis=(1, 2)) < args.max_ts_flag_frac
    ts_all = np.where(good_ts)[0]
    if len(ts_all) == 0:
        raise RuntimeError("no good timestamps")
    ts = ts_all[:: max(1, len(ts_all) // args.n_time_sample)]
    targets = cross if len(cross) <= args.n_targets else rng.choice(cross, args.n_targets, replace=False)
    log(f"sampling {len(ts)} timestamps x {len(targets)} target baselines, K={args.k} uv-neighbours")

    # per-(baseline,channel) mean over good timestamps = the mean-fill predictor
    Vm = V.copy(); Vm[F] = np.nan
    mf_c = np.nanmean(Vm[ts_all], axis=0)        # complex mean-fill, [n_bl, n_chan]
    mf_a = np.nanmean(np.abs(Vm[ts_all]), axis=0)
    del Vm

    num_cx = den_cx = 0.0      # complex cross-baseline
    num_ax = den_ax = 0.0      # amplitude cross-baseline
    num_cf = num_af = 0.0      # within-baseline freq-interp (same denominators)
    noise_pair = []            # nearest-neighbour |V_a-V_b|^2/2 ~ thermal noise proxy
    t0 = time.time()
    for ti, t in enumerate(ts):
        uv = uvw[t, cross, :2]
        tree = cKDTree(uv)
        for b in targets:
            ci = np.searchsorted(cross, b)
            d, idx = tree.query(uv[ci], k=args.k + 1)
            nbr = cross[idx[1:]]                  # drop self
            Vt = V[t, b]; Ft = F[t, b]            # target spectrum [n_chan]
            Vn = V[t, nbr]; Fn = F[t, nbr]        # neighbour spectra [K, n_chan]
            w = (~Fn).astype(np.float32)
            cnt = w.sum(0)
            pred = np.where(cnt > 0, (Vn * w).sum(0) / np.maximum(cnt, 1), np.nan)
            ok = (~Ft) & (cnt > 0) & np.isfinite(mf_c[b])
            if not ok.any():
                continue
            vt = Vt[ok]; pc = pred[ok]
            num_cx += np.sum(np.abs(vt - pc) ** 2)
            den_cx += np.sum(np.abs(vt - mf_c[b][ok]) ** 2)
            at = np.abs(vt); pa = np.abs(pc)
            num_ax += np.sum((at - pa) ** 2)
            den_ax += np.sum((at - mf_a[b][ok]) ** 2)
            # within-baseline: predict each channel from its unflagged freq neighbours
            cc = np.where(ok)[0]
            for c in cc:
                lo = c - 1; hi = c + 1
                vals = [V[t, b, x] for x in (lo, hi) if 0 <= x < n_chan and not F[t, b, x]]
                if vals:
                    pf = np.mean(vals)
                    num_cf += abs(Vt[c] - pf) ** 2
                    num_af += (abs(Vt[c]) - abs(pf)) ** 2
                else:
                    num_cf += abs(Vt[c] - mf_c[b][c]) ** 2
                    num_af += (abs(Vt[c]) - mf_a[b][c]) ** 2
            # nearest-neighbour noise proxy on the closest unflagged neighbour
            j = nbr[0]
            both = (~Ft) & (~F[t, j])
            if both.any():
                noise_pair.append(np.mean(np.abs(Vt[both] - V[t, j][both]) ** 2) / 2)
        if ti % 20 == 0:
            log(f"  ts {ti+1}/{len(ts)}  ({(ti+1)/max(time.time()-t0,1e-6):.1f}/s)  "
                f"running cx-R2={r2(num_cx,den_cx):.3f}")

    r2_cx, r2_ax = r2(num_cx, den_cx), r2(num_ax, den_ax)
    r2_cf, r2_af = r2(num_cf, den_cx), r2(num_af, den_ax)
    noise = float(np.median(noise_pair)) if noise_pair else float('nan')
    log("\n==== CROSS-BASELINE RECOVERABILITY ====")
    log(f"window {freqs[0]:.0f}-{freqs[-1]:.0f} MHz   (R2 = fraction of mean-fill error-variance removed)")
    log(f"  complex-V   cross-baseline R2 = {r2_cx:.3f}   within-baseline(freq) R2 = {r2_cf:.3f}")
    log(f"  amplitude   cross-baseline R2 = {r2_ax:.3f}   within-baseline(freq) R2 = {r2_af:.3f}")
    log(f"  (R2<=0 -> no better than mean-fill;  R2->1 -> fully recoverable)")
    log(f"  thermal-noise proxy (nearest-pair) median |n|^2 = {noise:.4g}")

    # uv-coherence: |corr| of complex visibility time-series vs baseline uv-separation
    cmid = n_chan // 2
    Vt = V[:, :, cmid].copy(); Ft = F[:, :, cmid]
    Vt[Ft] = np.nan
    npair = args.n_pairs
    a = rng.choice(cross, npair); b = rng.choice(cross, npair)
    seps, corrs = [], []
    for i, j in zip(a, b):
        if i == j:
            continue
        va, vb = Vt[ts_all, i], Vt[ts_all, j]
        m = np.isfinite(va) & np.isfinite(vb)
        if m.sum() < 20:
            continue
        va, vb = va[m] - va[m].mean(), vb[m] - vb[m].mean()
        denom = np.sqrt((np.abs(va) ** 2).sum() * (np.abs(vb) ** 2).sum())
        if denom <= 0:
            continue
        corrs.append(abs((va * np.conj(vb)).sum()) / denom)
        seps.append(float(np.nanmean(np.linalg.norm(uvw[ts_all][:, i, :2] - uvw[ts_all][:, j, :2], axis=1))))
    seps, corrs = np.array(seps), np.array(corrs)
    log("\n  uv-coherence |corr(V_i,V_j)| vs |Δuv| (metres):")
    edges = np.percentile(seps, [0, 20, 40, 60, 80, 100]) if len(seps) else np.zeros(6)
    for k in range(len(edges) - 1):
        m = (seps >= edges[k]) & (seps <= edges[k + 1])
        if m.any():
            log(f"    |Δuv| {edges[k]:8.1f}-{edges[k+1]:8.1f} m   |corr| {corrs[m].mean():.3f}  (n={m.sum()})")

    if args.out:
        np.savez(args.out, r2_complex_cross=r2_cx, r2_amp_cross=r2_ax,
                 r2_complex_freq=r2_cf, r2_amp_freq=r2_af, noise_proxy=noise,
                 uv_sep=seps, uv_corr=corrs, freqs=freqs)
        log(f"\nsaved -> {args.out}")
    log("RESULTLINE\t%.4f\t%.4f\t%.4f\t%.4f\t%.4g" % (r2_cx, r2_ax, r2_cf, r2_af, noise))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--column', default='DATA')
    ap.add_argument('--field', type=int, default=None)
    ap.add_argument('--freq-min', type=float, default=1300.0)
    ap.add_argument('--freq-max', type=float, default=1370.0)
    ap.add_argument('--n-chan', type=int, default=32)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--n-targets', type=int, default=200)
    ap.add_argument('--n-time-sample', type=int, default=300)
    ap.add_argument('--n-pairs', type=int, default=3000)
    ap.add_argument('--max-ts-flag-frac', type=float, default=0.95)
    ap.add_argument('--out', default=None)
    main(ap.parse_args())
