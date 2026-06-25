import argparse
import time
import numpy as np
import h5py
from pathlib import Path
from casacore.tables import table
from scipy.ndimage import uniform_filter1d
from skimage.transform import resize
from rfi_bands import LBAND_PERSISTENT_MHZ


def divisive_norm(waterfall, flagged, smooth_bins=64):
    norm = waterfall.copy()
    divisor = np.ones_like(waterfall)
    n_time, n_chan = waterfall.shape
    idx = np.arange(n_chan)
    for t in range(n_time):
        row = waterfall[t].copy()
        row[flagged[t]] = np.nan
        valid = ~np.isnan(row)
        if valid.sum() < smooth_bins:
            continue
        filled = row.copy()
        filled[~valid] = np.interp(idx[~valid], idx[valid], row[valid])
        smoothed = uniform_filter1d(filled, size=smooth_bins, mode='nearest')
        smoothed = np.clip(smoothed, 1e-6, None)
        norm[t] = waterfall[t] / smoothed
        divisor[t] = smoothed
    return norm, divisor


def resize_hw(arr, h, w, order=1):
    return resize(arr, (h, w), order=order, mode='edge',
                  anti_aliasing=(order > 0), preserve_range=True).astype(np.float32)


def resize_phase_hw(ph, h, w):
    return np.arctan2(resize_hw(np.sin(ph), h, w), resize_hw(np.cos(ph), h, w))


def good_runs(good_ts, min_run):
    runs = []
    n = len(good_ts)
    i = 0
    while i < n:
        if good_ts[i]:
            j = i
            while j < n and good_ts[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def windows_in(lo, hi, win, stride):
    out = []
    t = lo
    while t + win <= hi:
        out.append((t, t + win))
        t += stride
    if not out and hi - lo >= win // 2:
        out.append((lo, hi))
    return out


# each variant: (name, img_size, freq_tiles, time_win, max_flag) — how to slice a baseline.
#   freq_tiles: 1 = whole band; >1 = split band into N freq sub-bands
#   time_win:   None = use whole run (resize to img_size); int = slide time windows of that width
def variant_specs(sz):
    return {
        'v1_upsample512':   dict(sz=sz, freq_tiles=1, time_win=None, max_flag=0.5),
        'v2_time256':       dict(sz=256, freq_tiles=1, time_win=256, max_flag=0.5),
        'v3_freqtiled512':  dict(sz=sz, freq_tiles=2, time_win=None, max_flag=0.5),
        'v4_relaxed512':    dict(sz=sz, freq_tiles=1, time_win=None, max_flag=0.7),
        'v5_all512':        dict(sz=sz, freq_tiles=1, time_win=None, max_flag=0.85),
    }


def emit_dataset(path, spec, amp, phase, flagged, runs, cross_bls, ant1_bl, ant2_bl,
                 freqs, chan_lo, n_chan, is_test, smooth_bins):
    sz = spec['sz']; ft = spec['freq_tiles']; tw = spec['time_win']; mf = spec['max_flag']
    freq_edges = np.linspace(0, n_chan, ft + 1).astype(int)
    samples = []   # (bl, lo, hi, f0, f1)
    for lo, hi in runs:
        wins = windows_in(lo, hi, tw, tw // 2) if tw else [(lo, hi)]
        for bl in cross_bls:
            for (t0, t1) in wins:
                for fi in range(ft):
                    f0, f1 = freq_edges[fi], freq_edges[fi + 1]
                    fm = flagged[t0:t1, bl, f0:f1]
                    if fm.mean() <= mf:
                        samples.append((int(bl), t0, t1, f0, f1))
    if not samples:
        print(f"  {path.name}: 0 samples (max_flag={mf})", flush=True)
        return 0
    cap = len(samples)
    with h5py.File(path, 'w') as hf:
        def mk(name, dt, last=None):
            sh = (cap,) if last is None else (cap, last, last)
            return hf.create_dataset(name, shape=sh, dtype=dt,
                                     chunks=(1, last, last) if last else None)
        d_ds = mk('data', np.float32, sz); p_ds = mk('phase', np.float32, sz)
        fl_ds = mk('flags', np.float32, sz); dv_ds = mk('dn_divisor', np.float32, sz)
        bl_ds = mk('baseline_id', np.int32); a1 = mk('ant1', np.int32); a2 = mk('ant2', np.int32)
        tlo = mk('time_lo', np.int32); flo = mk('freq_lo', np.int32)
        ntd = mk('native_n_time', np.int32); ncd = mk('native_n_chan', np.int32)
        spl = mk('split', np.int32)   # 0 train, 1 test
        fmn = mk('freq_min_patch', np.float32); fmx = mk('freq_max_patch', np.float32)
        for k, (bl, t0, t1, f0, f1) in enumerate(samples):
            wf = amp[t0:t1, bl, f0:f1]; ph = phase[t0:t1, bl, f0:f1]
            fm = flagged[t0:t1, bl, f0:f1]
            wn, wd = divisive_norm(wf, fm, smooth_bins=min(smooth_bins, (f1 - f0) // 2))
            d_ds[k] = resize_hw(wn, sz, sz); p_ds[k] = resize_phase_hw(ph, sz, sz)
            fl_ds[k] = (resize_hw(fm.astype(np.float32), sz, sz) > 0.5).astype(np.float32)
            dv_ds[k] = resize_hw(wd, sz, sz)
            bl_ds[k] = bl; a1[k] = int(ant1_bl[bl]); a2[k] = int(ant2_bl[bl])
            tlo[k] = t0; flo[k] = f0; ntd[k] = t1 - t0; ncd[k] = f1 - f0; spl[k] = int(is_test[bl])
            fmn[k] = float(freqs[f0]); fmx[k] = float(freqs[min(f1 - 1, n_chan - 1)])
        hf.attrs['freq_min_mhz'] = float(freqs[0]); hf.attrs['freq_max_mhz'] = float(freqs[-1])
        hf.attrs['n_time'] = sz; hf.attrs['n_freq'] = sz; hf.attrs['img_size'] = sz
        hf.attrs['n_baselines'] = cap; hf.attrs['full_n_chan'] = n_chan; hf.attrs['chan_lo'] = chan_lo
    n_test = int(sum(is_test[bl] for bl, *_ in samples))
    print(f"  {path.name}: {cap} samples ({cap - n_test} train, {n_test} test)  "
          f"sz={sz} ft={ft} tw={tw} maxflag={mf}", flush=True)
    return cap


def main(args):
    t0 = time.time()
    ms = table(args.ms, readonly=True)
    sw = table(args.ms + '/SPECTRAL_WINDOW'); freqs_full = sw.getcol('CHAN_FREQ')[0] / 1e6; sw.close()
    cm = (freqs_full >= args.freq_min) & (freqs_full <= args.freq_max)
    ci = np.where(cm)[0]; chan_lo, chan_hi = int(ci[0]), int(ci[-1]) + 1
    n_chan = chan_hi - chan_lo; freqs = freqs_full[chan_lo:chan_hi]
    persist = np.zeros(n_chan, bool)
    for a, b in LBAND_PERSISTENT_MHZ:
        persist |= (freqs >= a) & (freqs <= b)

    times = ms.getcol('TIME'); ant1 = ms.getcol('ANTENNA1'); ant2 = ms.getcol('ANTENNA2')
    n_row = ms.nrows(); chunk = 50000
    amp = np.empty((n_row, n_chan), np.float32); phase = np.empty((n_row, n_chan), np.float32)
    flagged = np.empty((n_row, n_chan), bool)
    print(f"reading {args.column} ({n_row} rows, {n_chan} ch)...", flush=True)
    for s in range(0, n_row, chunk):
        e = min(s + chunk, n_row)
        d = ms.getcol(args.column, startrow=s, nrow=e - s)[:, chan_lo:chan_hi, :]
        f = ms.getcol('FLAG', startrow=s, nrow=e - s)[:, chan_lo:chan_hi, :]
        amp[s:e] = np.abs(d).mean(axis=2); phase[s:e] = np.angle(d.mean(axis=2))
        flagged[s:e] = f.any(axis=2)
        if s == 0 or (s // chunk) % 10 == 0:
            print(f"  rows {e}/{n_row}", flush=True)
    ms.close()
    print(f"read done ({time.time() - t0:.0f}s)", flush=True)

    flagged[:, persist] = True
    tt = np.unique(times); ntime = len(tt); nbl = amp.shape[0] // ntime
    amp = amp[:ntime * nbl].reshape(ntime, nbl, n_chan)
    phase = phase[:ntime * nbl].reshape(ntime, nbl, n_chan)
    flagged = flagged[:ntime * nbl].reshape(ntime, nbl, n_chan)
    ant1_bl = ant1[:nbl]; ant2_bl = ant2[:nbl]; autocorr = ant1_bl == ant2_bl
    cross_bls = np.where(~autocorr)[0]

    good = flagged.mean(axis=(1, 2)) < args.max_ts_flag_frac
    runs = good_runs(good, args.min_run)
    print(f"timestamps {ntime}, good {int(good.sum())}, runs {[(l, h - l) for l, h in runs]}", flush=True)
    if not runs:
        raise RuntimeError("no good time runs")

    rng = np.random.default_rng(args.split_seed)
    is_test = np.zeros(nbl, bool)
    test_bls = rng.choice(cross_bls, size=int(len(cross_bls) * args.test_frac), replace=False)
    is_test[test_bls] = True
    print(f"baselines: {len(cross_bls)} cross, {len(test_bls)} held out for test", flush=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    specs = variant_specs(args.img_size)
    if args.only:
        keep = set(args.only.split(','))
        specs = {n: s for n, s in specs.items() if n in keep}
        if not specs:
            raise RuntimeError(f"--only {args.only} matched no variant of {list(variant_specs(args.img_size))}")
    for name, spec in specs.items():
        emit_dataset(out / f'{name}.h5', spec, amp, phase, flagged, runs, cross_bls,
                     ant1_bl, ant2_bl, freqs, chan_lo, n_chan, is_test, args.smooth_bins)
    print(f"all variants -> {out}  ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ms', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--column', default='DATA')
    p.add_argument('--freq-min', type=float, default=900.0)
    p.add_argument('--freq-max', type=float, default=1650.0)
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--max-ts-flag-frac', type=float, default=0.95)
    p.add_argument('--min-run', type=int, default=64)
    p.add_argument('--test-frac', type=float, default=0.15)
    p.add_argument('--split-seed', type=int, default=1234)
    p.add_argument('--smooth-bins', type=int, default=64)
    p.add_argument('--only', default=None, help='comma-separated variant names to build (default all)')
    main(p.parse_args())
