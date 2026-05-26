import argparse
import numpy as np
import h5py
from pathlib import Path
from casacore.tables import table
from scipy.ndimage import uniform_filter1d


def divisive_norm(waterfall, flagged, smooth_bins=64):
    norm = waterfall.copy()
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
    return norm


def extract_patches(waterfall, flagged, pt, pf, st, sf, max_flag_frac, max_patches):
    n_time, n_chan = waterfall.shape
    patches, freq_offsets = [], []
    t = 0
    while t + pt <= n_time:
        f = 0
        while f + pf <= n_chan:
            if flagged[t:t+pt, f:f+pf].mean() <= max_flag_frac:
                patches.append(waterfall[t:t+pt, f:f+pf].copy())
                freq_offsets.append(f)
            f += sf
            if len(patches) >= max_patches:
                return patches, freq_offsets
        t += st
    return patches, freq_offsets


def main(args):
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

    n_row = ms.nrows()
    chunk = 50000
    amp     = np.empty((n_row, n_chan), dtype=np.float32)
    flagged = np.empty((n_row, n_chan), dtype=bool)

    print(f"reading {col} in chunks ({n_row} rows, {n_chan} channels)...")
    for start in range(0, n_row, chunk):
        end = min(start + chunk, n_row)
        d = ms.getcol(col,    startrow=start, nrow=end - start)[:, chan_lo:chan_hi, :]
        f = ms.getcol('FLAG', startrow=start, nrow=end - start)[:, chan_lo:chan_hi, :]
        amp[start:end]     = np.abs(d).mean(axis=2).astype(np.float32)
        flagged[start:end] = f.any(axis=2)
        del d, f
        if start == 0 or (start // chunk) % 5 == 0:
            print(f"  rows {end}/{n_row}")

    ms.close()
    print("read complete")

    unique_times = np.unique(times)
    n_time     = len(unique_times)
    n_baseline = amp.shape[0] // n_time

    amp     = amp[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    flagged = flagged[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    ant1_bl  = ant1[:n_baseline]
    ant2_bl  = ant2[:n_baseline]
    autocorr = ant1_bl == ant2_bl

    freq_min = float(freqs[0])
    freq_max = float(freqs[-1])
    pt, pf = args.patch_time, args.patch_freq
    st, sf = args.stride_time, args.stride_freq

    wf_sum   = np.zeros((n_time, n_chan), dtype=np.float64)
    wf_count = np.zeros((n_time, n_chan), dtype=np.int32)

    baselines_used = 0
    baselines_skipped = 0
    n_cross = int((~autocorr).sum())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # write patches directly to HDF5 as we go — avoids holding all patches in RAM
    all_offsets = []
    n_patches_written = 0

    with h5py.File(out, 'w') as hf:
        max_patches_est = n_cross * args.max_patches_per_bl
        clean_ds      = hf.create_dataset('clean',          shape=(max_patches_est, pt, pf),
                                          maxshape=(None, pt, pf), dtype=np.float32,
                                          chunks=(1, pt, pf))
        freq_min_ds   = hf.create_dataset('freq_min_patch', shape=(max_patches_est,),
                                          maxshape=(None,), dtype=np.float32)
        freq_max_ds   = hf.create_dataset('freq_max_patch', shape=(max_patches_est,),
                                          maxshape=(None,), dtype=np.float32)

        cross_idx = 0
        for bl in range(n_baseline):
            if autocorr[bl]:
                continue

            cross_idx += 1
            if cross_idx % 200 == 0 or cross_idx == 1:
                print(f"  baseline {cross_idx}/{n_cross}  patches written: {n_patches_written}")

            wf = amp[:, bl, :]
            fm = flagged[:, bl, :]

            if fm.mean() > args.max_bl_flag_frac:
                baselines_skipped += 1
                continue

            valid = ~fm
            wf_sum[valid]   += wf[valid]
            wf_count[valid] += 1

            wf_norm = divisive_norm(wf, fm, smooth_bins=args.smooth_bins)

            patches, offsets = extract_patches(
                wf_norm, fm, pt, pf, st, sf,
                args.max_flag_frac, args.max_patches_per_bl
            )

            if not patches:
                baselines_skipped += 1
                continue

            n = len(patches)
            for i, (p, o) in enumerate(zip(patches, offsets)):
                idx = n_patches_written + i
                clean_ds[idx]    = p
                freq_min_ds[idx] = freqs[o]
                freq_max_ds[idx] = freqs[min(o + pf - 1, n_chan - 1)]

            all_offsets.extend(offsets)
            n_patches_written += n
            baselines_used += 1

        # trim datasets to actual size
        clean_ds.resize(n_patches_written, axis=0)
        freq_min_ds.resize(n_patches_written, axis=0)
        freq_max_ds.resize(n_patches_written, axis=0)

        hf.attrs['freq_min_mhz'] = freq_min
        hf.attrs['freq_max_mhz'] = freq_max
        hf.attrs['n_time']       = pt
        hf.attrs['n_freq']       = pf
        hf.attrs['n_patches']    = n_patches_written

    if n_patches_written == 0:
        raise RuntimeError("no patches extracted — check flag fraction thresholds")

    print(f"column         : {col}")
    print(f"freq range     : {freq_min:.1f}-{freq_max:.1f} MHz  ({n_chan} channels)")
    print(f"baselines used : {baselines_used}  skipped: {baselines_skipped}  autocorr: {autocorr.sum()}")
    print(f"patches total  : {n_patches_written}  shape: ({pt}, {pf})")

    if args.waterfall_out:
        wf_avg   = np.where(wf_count > 0, wf_sum / wf_count, 0.0).astype(np.float32)
        flag_avg = (wf_count == 0)
        np.save(args.waterfall_out + '.npy',       wf_avg)
        np.save(args.waterfall_out + '.meta.npy',  np.array([freq_min, freq_max]))
        np.save(args.waterfall_out + '.flags.npy', flag_avg)
        print(f"waterfall      -> {args.waterfall_out}.npy")

    print(f"saved -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms',                  required=True)
    parser.add_argument('--output',              required=True)
    parser.add_argument('--waterfall-out',       default=None)
    parser.add_argument('--freq-min',            type=float, default=900.0)
    parser.add_argument('--freq-max',            type=float, default=1650.0)
    parser.add_argument('--patch-time',          type=int,   default=256)
    parser.add_argument('--patch-freq',          type=int,   default=256)
    parser.add_argument('--stride-time',         type=int,   default=64)
    parser.add_argument('--stride-freq',         type=int,   default=64)
    parser.add_argument('--max-patches-per-bl',  type=int,   default=50)
    parser.add_argument('--max-flag-frac',       type=float, default=0.5)
    parser.add_argument('--max-bl-flag-frac',    type=float, default=0.8)
    parser.add_argument('--smooth-bins',         type=int,   default=64)
    main(parser.parse_args())
