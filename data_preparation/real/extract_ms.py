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

    # drop near-fully-flagged timestamps — they carry no signal and inflate every
    # baseline's flag fraction. ~half this MS's dumps are dead (setup/slew scans).
    ts_frac = flagged.mean(axis=(1, 2))
    good_ts = ts_frac < args.max_ts_flag_frac
    n_bad_ts = int((~good_ts).sum())
    amp     = amp[good_ts]
    phase   = phase[good_ts]
    flagged = flagged[good_ts]
    n_time_good = int(good_ts.sum())
    print(f"timestamps: {n_time} total, dropped {n_bad_ts} bad (>{args.max_ts_flag_frac:.2f}), "
          f"kept {n_time_good}", flush=True)

    freq_min, freq_max = float(freqs[0]), float(freqs[-1])
    sz = args.img_size
    n_cross = int((~autocorr).sum())

    bl_frac = flagged[:, ~autocorr, :].mean(axis=(0, 2))
    print(f"per-baseline flag frac (force_persistent={args.force_persistent}, good-TS): "
          f"min {bl_frac.min():.3f}  p25 {np.percentile(bl_frac, 25):.3f}  "
          f"p50 {np.percentile(bl_frac, 50):.3f}  mean {bl_frac.mean():.3f}  | "
          f"<{args.max_bl_flag_frac:.2f}: {(bl_frac <= args.max_bl_flag_frac).sum()}/{n_cross}",
          flush=True)
    if (bl_frac <= args.max_bl_flag_frac).sum() == 0:
        raise RuntimeError(
            f"no baselines <= max_bl_flag_frac={args.max_bl_flag_frac} "
            f"(cleanest is {bl_frac.min():.3f}); raise --max-bl-flag-frac")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    baselines_skipped = 0

    with h5py.File(out, 'w') as hf:
        def mk(name, dtype, last=None):
            shape = (n_cross,) if last is None else (n_cross, last, last)
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

        cross_idx = 0
        for bl in range(n_baseline):
            if autocorr[bl]:
                continue
            cross_idx += 1
            if cross_idx % 200 == 0 or cross_idx == 1:
                rate = cross_idx / max(time.time() - t_start, 1e-6)
                print(f"  baseline {cross_idx}/{n_cross}  written: {n_written}  "
                      f"({rate:.1f} bl/s)", flush=True)

            wf = amp[:, bl, :]
            ph = phase[:, bl, :]
            fm = flagged[:, bl, :]

            if fm.mean() > args.max_bl_flag_frac:
                baselines_skipped += 1
                continue

            wf_norm, wf_div = divisive_norm(wf, fm, smooth_bins=args.smooth_bins)

            amp_ds[n_written]     = resize_to(wf_norm, sz, order=1)
            phase_ds[n_written]   = resize_to(ph, sz, order=1)
            flags_ds[n_written]   = (resize_to(fm.astype(np.float32), sz, order=1) > 0.5).astype(np.float32)
            divisor_ds[n_written] = resize_to(wf_div, sz, order=1)
            bl_id_ds[n_written]   = bl
            ant1_ds[n_written]    = int(ant1_bl[bl])
            ant2_ds[n_written]    = int(ant2_bl[bl])
            nchan_ds[n_written]   = n_chan
            ntime_ds[n_written]   = n_time_good
            n_written += 1

        for ds in (amp_ds, phase_ds, flags_ds, divisor_ds, bl_id_ds, ant1_ds, ant2_ds,
                   nchan_ds, ntime_ds):
            ds.resize(n_written, axis=0)

        hf.attrs['freq_min_mhz'] = freq_min
        hf.attrs['freq_max_mhz'] = freq_max
        hf.attrs['n_time']       = sz
        hf.attrs['n_freq']       = sz
        hf.attrs['img_size']     = sz
        hf.attrs['n_baselines']  = n_written
        hf.attrs['full_n_time']  = n_time_good
        hf.attrs['full_n_chan']  = n_chan
        hf.attrs['chan_lo']      = chan_lo
        hf.attrs['column']       = col

    if n_written == 0:
        raise RuntimeError("no baselines extracted — check max_bl_flag_frac")

    print(f"column         : {col}", flush=True)
    print(f"freq range     : {freq_min:.1f}-{freq_max:.1f} MHz  ({n_chan} channels -> {sz})", flush=True)
    print(f"native n_time  : {n_time_good} (good) -> {sz}", flush=True)
    print(f"baselines kept : {n_written}  skipped: {baselines_skipped}  autocorr: {autocorr.sum()}", flush=True)
    print(f"saved {n_written} per-baseline waterfalls ({sz}x{sz}) -> {out}", flush=True)


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
    parser.add_argument('--smooth-bins',      type=int,   default=64)
    parser.add_argument('--force-persistent', action='store_true', default=True)
    parser.add_argument('--no-force-persistent', dest='force_persistent', action='store_false')
    main(parser.parse_args())
