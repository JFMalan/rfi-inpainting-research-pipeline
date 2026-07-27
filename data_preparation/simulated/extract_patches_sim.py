import argparse
import time
import numpy as np
import h5py
from pathlib import Path
from casacore.tables import table
from scipy.ndimage import uniform_filter1d


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


def main(args):
    t_start = time.time()
    ms = table(args.ms, readonly=True)
    cols = ms.colnames()
    col = 'CORRECTED_DATA' if 'CORRECTED_DATA' in cols else 'DATA'
    try:
        ms.getcell(col, 0)
    except Exception:
        col = 'DATA'

    freqs_tab = table(args.ms + '/SPECTRAL_WINDOW')
    freqs_full = freqs_tab.getcol('CHAN_FREQ')[0] / 1e6
    freqs_tab.close()

    chan_mask = (freqs_full >= args.freq_min) & (freqs_full <= args.freq_max)
    chan_indices = np.where(chan_mask)[0]
    chan_lo  = int(chan_indices[0])
    chan_hi  = int(chan_indices[-1]) + 1
    n_chan   = chan_hi - chan_lo
    freqs    = freqs_full[chan_lo:chan_hi]

    times = ms.getcol('TIME')
    ant1  = ms.getcol('ANTENNA1')
    ant2  = ms.getcol('ANTENNA2')

    # noise-free target: CLEAN_DATA is the pre-noise snapshot written by add_noise.py.
    # Both clean amplitude and clean phase become target fields; the noisy column stays
    # the input/conditioning and its divisor is shared (renormalisation contract).
    clean_col = 'CLEAN_DATA' if 'CLEAN_DATA' in cols else None

    n_row = ms.nrows()
    chunk = 50000
    amp     = np.empty((n_row, n_chan), dtype=np.float32)
    phase   = np.empty((n_row, n_chan), dtype=np.float32)
    flagged = np.empty((n_row, n_chan), dtype=bool)
    if clean_col:
        amp_t   = np.empty((n_row, n_chan), dtype=np.float32)
        phase_t = np.empty((n_row, n_chan), dtype=np.float32)

    print(f"reading {col}{' + ' + clean_col if clean_col else ''} in chunks "
          f"({n_row} rows, {n_chan} channels)...", flush=True)
    for start in range(0, n_row, chunk):
        end = min(start + chunk, n_row)
        d = ms.getcol(col,    startrow=start, nrow=end - start)[:, chan_lo:chan_hi, :]
        f = ms.getcol('FLAG', startrow=start, nrow=end - start)[:, chan_lo:chan_hi, :]
        amp[start:end]     = np.abs(d).mean(axis=2).astype(np.float32)
        phase[start:end]   = np.angle(d.mean(axis=2)).astype(np.float32)
        flagged[start:end] = f.any(axis=2)
        del d, f
        if clean_col:
            c = ms.getcol(clean_col, startrow=start, nrow=end - start)[:, chan_lo:chan_hi, :]
            amp_t[start:end]   = np.abs(c).mean(axis=2).astype(np.float32)
            phase_t[start:end] = np.angle(c.mean(axis=2)).astype(np.float32)
            del c
        if start == 0 or (start // chunk) % 5 == 0:
            print(f"  rows {end}/{n_row}", flush=True)

    ms.close()
    print(f"read complete ({time.time() - t_start:.0f}s)", flush=True)

    unique_times = np.unique(times)
    n_time     = len(unique_times)
    n_baseline = amp.shape[0] // n_time

    amp     = amp[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    phase   = phase[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    flagged = flagged[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    if clean_col:
        amp_t   = amp_t[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
        phase_t = phase_t[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    ant1_bl  = ant1[:n_baseline]
    ant2_bl  = ant2[:n_baseline]
    autocorr = ant1_bl == ant2_bl

    freq_min = float(freqs[0])
    freq_max = float(freqs[-1])

    wf_sum   = np.zeros((n_time, n_chan), dtype=np.float64)
    wf_count = np.zeros((n_time, n_chan), dtype=np.int32)

    baselines_used = 0
    baselines_skipped = 0
    n_cross = int((~autocorr).sum())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0

    with h5py.File(out, 'w') as hf:
        def mk(name, dtype, shape):
            return hf.create_dataset(name, shape=(n_cross,) + shape,
                                     maxshape=(None,) + shape, dtype=dtype,
                                     chunks=(1,) + shape if shape else None)

        clean_ds   = mk('clean',      np.float32, (n_time, n_chan))
        divisor_ds = mk('dn_divisor', np.float32, (n_time, n_chan))
        phase_ds   = mk('phase',      np.float32, (n_time, n_chan))
        bl_id_ds   = mk('baseline_id', np.int32, ())
        ant1_ds    = mk('ant1', np.int32, ())
        ant2_ds    = mk('ant2', np.int32, ())
        nchan_ds   = mk('native_n_chan', np.int32, ())
        ntime_ds   = mk('native_n_time', np.int32, ())
        if clean_col:
            amp_t_ds   = mk('amp_target',   np.float32, (n_time, n_chan))
            phase_t_ds = mk('phase_target', np.float32, (n_time, n_chan))

        for bl in range(n_baseline):
            if autocorr[bl]:
                continue

            if (bl + 1) % 200 == 0 or bl == 0:
                rate = (n_written + 1) / max(time.time() - t_start, 1e-6)
                print(f"  baseline {bl + 1}/{n_baseline}  written: {n_written}  "
                      f"({rate:.1f} bl/s)", flush=True)

            wf = amp[:, bl, :]
            ph = phase[:, bl, :]
            fm = flagged[:, bl, :]

            if fm.mean() > args.max_bl_flag_frac:
                baselines_skipped += 1
                continue

            valid = ~fm
            wf_sum[valid]   += wf[valid]
            wf_count[valid] += 1

            wf_norm, wf_div = divisive_norm(wf, fm, smooth_bins=args.smooth_bins)

            clean_ds[n_written]   = wf_norm.astype(np.float32)
            divisor_ds[n_written] = wf_div.astype(np.float32)
            phase_ds[n_written]   = ph.astype(np.float32)
            bl_id_ds[n_written]   = bl
            ant1_ds[n_written]    = int(ant1_bl[bl])
            ant2_ds[n_written]    = int(ant2_bl[bl])
            nchan_ds[n_written]   = n_chan
            ntime_ds[n_written]   = n_time
            if clean_col:
                # clean amplitude renormalised by the NOISY divisor (shared contract)
                amp_t_ds[n_written]   = (amp_t[:, bl, :] / wf_div).astype(np.float32)
                phase_t_ds[n_written] = phase_t[:, bl, :].astype(np.float32)
            n_written += 1
            baselines_used += 1

        all_ds = [clean_ds, divisor_ds, phase_ds, bl_id_ds, ant1_ds, ant2_ds,
                  nchan_ds, ntime_ds]
        if clean_col:
            all_ds += [amp_t_ds, phase_t_ds]
        for ds in all_ds:
            ds.resize(n_written, axis=0)

        hf.attrs['freq_min_mhz'] = freq_min
        hf.attrs['freq_max_mhz'] = freq_max
        hf.attrs['n_baselines']  = n_written
        hf.attrs['full_n_time']  = n_time
        hf.attrs['full_n_chan']  = n_chan
        hf.attrs['chan_lo']      = chan_lo
        hf.attrs['clean_target'] = bool(clean_col)

    if n_written == 0:
        raise RuntimeError("no baselines extracted — check max_bl_flag_frac")

    print(f"column         : {col}  clean target: {clean_col or 'none'}", flush=True)
    print(f"freq range     : {freq_min:.1f}-{freq_max:.1f} MHz  ({n_chan} native channels)", flush=True)
    print(f"native n_time  : {n_time}", flush=True)
    print(f"baselines used : {baselines_used}  skipped: {baselines_skipped}  autocorr: {autocorr.sum()}", flush=True)
    print(f"saved {n_written} native per-baseline waterfalls ({n_time}x{n_chan}) -> {out}", flush=True)

    if args.waterfall_out:
        wf_avg   = np.where(wf_count > 0, wf_sum / wf_count, 0.0).astype(np.float32)
        flag_avg = (wf_count == 0)
        np.save(args.waterfall_out + '.npy',       wf_avg)
        np.save(args.waterfall_out + '.meta.npy',  np.array([freq_min, freq_max]))
        np.save(args.waterfall_out + '.flags.npy', flag_avg)
        print(f"waterfall      -> {args.waterfall_out}.npy", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms',               required=True)
    parser.add_argument('--output',           required=True)
    parser.add_argument('--waterfall-out',    default=None)
    parser.add_argument('--freq-min',         type=float, default=900.0)
    parser.add_argument('--freq-max',         type=float, default=1650.0)
    parser.add_argument('--max-bl-flag-frac', type=float, default=0.8)
    parser.add_argument('--smooth-bins',      type=int,   default=64)
    main(parser.parse_args())
