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


def resize_to(arr, size, order=1):
    return resize(arr, (size, size), order=order, mode='edge',
                  anti_aliasing=(order > 0), preserve_range=True).astype(np.float32)


def resize_phase(ph, size):
    return np.arctan2(resize_to(np.sin(ph), size), resize_to(np.cos(ph), size))


def main(args):
    t_start = time.time()
    ms = table(args.ms, readonly=True)
    if args.field is not None:
        ms = ms.query(f"FIELD_ID == {args.field}")

    cols = ms.colnames()
    col = args.column if args.column is not None else 'DATA'
    if col not in cols:
        raise RuntimeError(f"column {col} not in MS ({cols})")

    freqs_tab = table(args.ms + '/SPECTRAL_WINDOW')
    freqs_full = freqs_tab.getcol('CHAN_FREQ')[0] / 1e6
    freqs_tab.close()

    chan_mask = (freqs_full >= args.freq_min) & (freqs_full <= args.freq_max)
    chan_idx = np.where(chan_mask)[0]
    chan_lo, chan_hi = int(chan_idx[0]), int(chan_idx[-1]) + 1
    n_chan = chan_hi - chan_lo
    freqs = freqs_full[chan_lo:chan_hi]
    persist_band = np.zeros(n_chan, dtype=bool)
    for flo, fhi in LBAND_PERSISTENT_MHZ:
        persist_band |= (freqs >= flo) & (freqs <= fhi)

    times = ms.getcol('TIME')
    ant1 = ms.getcol('ANTENNA1')
    ant2 = ms.getcol('ANTENNA2')

    n_row = ms.nrows()
    chunk = 50000
    amp = np.empty((n_row, n_chan), dtype=np.float32)
    phase = np.empty((n_row, n_chan), dtype=np.float32)
    flagged = np.empty((n_row, n_chan), dtype=bool)

    print(f"reading {col} in chunks ({n_row} rows, {n_chan} channels)...", flush=True)
    for start in range(0, n_row, chunk):
        end = min(start + chunk, n_row)
        d = ms.getcol(col,    startrow=start, nrow=end - start)[:, chan_lo:chan_hi, :]
        f = ms.getcol('FLAG', startrow=start, nrow=end - start)[:, chan_lo:chan_hi, :]
        amp[start:end]     = np.abs(d).mean(axis=2).astype(np.float32)
        phase[start:end]   = np.angle(d.mean(axis=2)).astype(np.float32)
        flagged[start:end] = f.any(axis=2)
        del d, f
        if start == 0 or (start // chunk) % 5 == 0:
            print(f"  rows {end}/{n_row}", flush=True)
    ms.close()
    print(f"read complete ({time.time() - t_start:.0f}s)", flush=True)

    if args.force_persistent:
        flagged[:, persist_band] = True

    unique_times = np.unique(times)
    n_time = len(unique_times)
    n_baseline = amp.shape[0] // n_time
    amp     = amp[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    phase   = phase[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    flagged = flagged[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    ant1_bl = ant1[:n_baseline]
    ant2_bl = ant2[:n_baseline]
    autocorr = ant1_bl == ant2_bl

    # find contiguous runs of good timestamps. near-fully-flagged dumps (setup/slew)
    # split the observation into separate good time-blocks; each block is extracted
    # as its own contiguous waterfall so the time axis stays seam-free.
    ts_frac = flagged.mean(axis=(1, 2))
    good_ts = ts_frac < args.max_ts_flag_frac
    runs = []
    i = 0
    while i < n_time:
        if good_ts[i]:
            j = i
            while j < n_time and good_ts[j]:
                j += 1
            if (j - i) >= args.min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        raise RuntimeError(f"no contiguous good run >= {args.min_run} timestamps")
    print(f"timestamps: {n_time} total, {int(good_ts.sum())} good, "
          f"{len(runs)} contiguous run(s) >= {args.min_run}: "
          f"{[(lo, hi - lo) for lo, hi in runs]}", flush=True)

    freq_min, freq_max = float(freqs[0]), float(freqs[-1])
    sz = args.img_size
    n_cross = int((~autocorr).sum())
    cross_bls = np.where(~autocorr)[0]

    # per (baseline, run) flag fraction, to size the cut and the output
    fracs = []
    for lo, hi in runs:
        fracs.append(flagged[lo:hi][:, ~autocorr, :].mean(axis=(0, 2)))
    fracs = np.concatenate(fracs)
    n_units = len(runs) * n_cross
    print(f"per (baseline,run) flag frac (force_persistent={args.force_persistent}): "
          f"min {fracs.min():.3f}  p25 {np.percentile(fracs, 25):.3f}  "
          f"p50 {np.percentile(fracs, 50):.3f}  mean {fracs.mean():.3f}  | "
          f"<{args.max_bl_flag_frac:.2f}: {(fracs <= args.max_bl_flag_frac).sum()}/{n_units}",
          flush=True)
    if (fracs <= args.max_bl_flag_frac).sum() == 0:
        raise RuntimeError(
            f"no (baseline,run) <= max_bl_flag_frac={args.max_bl_flag_frac} "
            f"(cleanest is {fracs.min():.3f}); raise --max-bl-flag-frac")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    baselines_skipped = 0

    cap = n_units
    with h5py.File(out, 'w') as hf:
        def mk(name, dtype, last=None):
            shape = (cap,) if last is None else (cap, last, last)
            maxshape = (None,) if last is None else (None, last, last)
            chunks = None if last is None else (1, last, last)
            return hf.create_dataset(name, shape=shape, maxshape=maxshape,
                                     dtype=dtype, chunks=chunks)

        amp_ds     = mk('data',       np.float32, sz)
        phase_ds   = mk('phase',      np.float32, sz)
        flags_ds   = mk('flags',      np.float32, sz)
        divisor_ds = mk('dn_divisor', np.float32, sz)
        bl_id_ds   = mk('baseline_id', np.int32)
        ant1_ds    = mk('ant1', np.int32)
        ant2_ds    = mk('ant2', np.int32)
        nchan_ds   = mk('native_n_chan', np.int32)
        ntime_ds   = mk('native_n_time', np.int32)
        tlo_ds     = mk('time_lo', np.int32)
        run_ds     = mk('run_id', np.int32)

        unit = 0
        for r, (lo, hi) in enumerate(runs):
            run_len = hi - lo
            for bl in cross_bls:
                unit += 1
                if unit % 500 == 0 or unit == 1:
                    rate = unit / max(time.time() - t_start, 1e-6)
                    print(f"  run {r} unit {unit}/{n_units}  written: {n_written}  "
                          f"({rate:.1f}/s)", flush=True)

                wf = amp[lo:hi, bl, :]
                ph = phase[lo:hi, bl, :]
                fm = flagged[lo:hi, bl, :]

                if fm.mean() > args.max_bl_flag_frac:
                    baselines_skipped += 1
                    continue

                wf_norm, wf_div = divisive_norm(wf, fm, smooth_bins=args.smooth_bins)

                amp_ds[n_written]     = resize_to(wf_norm, sz, order=1)
                phase_ds[n_written]   = resize_phase(ph, sz)
                flags_ds[n_written]   = (resize_to(fm.astype(np.float32), sz, order=1) > 0.5).astype(np.float32)
                divisor_ds[n_written] = resize_to(wf_div, sz, order=1)
                bl_id_ds[n_written]   = int(bl)
                ant1_ds[n_written]    = int(ant1_bl[bl])
                ant2_ds[n_written]    = int(ant2_bl[bl])
                nchan_ds[n_written]   = n_chan
                ntime_ds[n_written]   = run_len
                tlo_ds[n_written]     = lo
                run_ds[n_written]     = r
                n_written += 1

        for ds in (amp_ds, phase_ds, flags_ds, divisor_ds, bl_id_ds, ant1_ds, ant2_ds,
                   nchan_ds, ntime_ds, tlo_ds, run_ds):
            ds.resize(n_written, axis=0)

        hf.attrs['freq_min_mhz'] = freq_min
        hf.attrs['freq_max_mhz'] = freq_max
        hf.attrs['n_time']       = sz
        hf.attrs['n_freq']       = sz
        hf.attrs['img_size']     = sz
        hf.attrs['n_baselines']  = n_written
        hf.attrs['n_runs']       = len(runs)
        hf.attrs['full_n_chan']  = n_chan
        hf.attrs['chan_lo']      = chan_lo
        hf.attrs['column']       = col

    if n_written == 0:
        raise RuntimeError("no baselines extracted — check max_bl_flag_frac")

    print(f"column         : {col}", flush=True)
    print(f"freq range     : {freq_min:.1f}-{freq_max:.1f} MHz  ({n_chan} channels -> {sz})", flush=True)
    print(f"runs           : {[(lo, hi - lo) for lo, hi in runs]} -> resized to {sz}", flush=True)
    print(f"waterfalls kept: {n_written}  skipped: {baselines_skipped}  (of {n_units} baseline-run units)", flush=True)
    print(f"saved {n_written} per-(baseline,run) waterfalls ({sz}x{sz}) -> {out}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms',               required=True)
    parser.add_argument('--output',           required=True)
    parser.add_argument('--column',           default='DATA')
    parser.add_argument('--field',            type=int,   default=None)
    parser.add_argument('--freq-min',         type=float, default=900.0)
    parser.add_argument('--freq-max',         type=float, default=1650.0)
    parser.add_argument('--img-size',         type=int,   default=512)
    parser.add_argument('--max-bl-flag-frac', type=float, default=0.5)
    parser.add_argument('--max-ts-flag-frac', type=float, default=0.95)
    parser.add_argument('--min-run',          type=int,   default=64)
    parser.add_argument('--smooth-bins',      type=int,   default=64)
    parser.add_argument('--force-persistent', action='store_true', default=True)
    parser.add_argument('--no-force-persistent', dest='force_persistent', action='store_false')
    main(parser.parse_args())
