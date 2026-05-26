import argparse
import numpy as np
import h5py
from pathlib import Path
from casacore.tables import table
from scipy.ndimage import uniform_filter1d


def divisive_norm(waterfall, flagged, smooth_bins=64):
    # vectorised: operate on all time rows at once
    wf = waterfall.copy()
    wf[flagged] = np.nan

    # uniform_filter1d handles NaNs poorly, so use a masked cumsum approach:
    # for each row, interpolate over NaNs then smooth, then divide
    norm = waterfall.copy()
    half = smooth_bins // 2
    n_time, n_chan = waterfall.shape
    idx = np.arange(n_chan)

    for t in range(n_time):
        row = wf[t]
        valid = ~np.isnan(row)
        if valid.sum() < smooth_bins:
            continue
        # fill NaNs by interpolation so uniform_filter1d doesn't propagate them
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

    print(f"reading {col} column (channels {chan_lo}:{chan_hi} = {n_chan} ch)...")
    data  = ms.getcol(col,   startchan=chan_lo, nchans=n_chan)
    flags = ms.getcol('FLAG', startchan=chan_lo, nchans=n_chan)
    ms.close()
    print(f"read complete — shape {data.shape}")

    amp     = np.abs(data).mean(axis=2).astype(np.float32)
    flagged = flags.any(axis=2)
    del data, flags

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

    all_patches, all_offsets = [], []
    wf_sum   = np.zeros((n_time, n_chan), dtype=np.float64)
    wf_count = np.zeros((n_time, n_chan), dtype=np.int32)

    baselines_used = 0
    baselines_skipped = 0
    n_cross = int((~autocorr).sum())

    cross_idx = 0
    for bl in range(n_baseline):
        if autocorr[bl]:
            continue

        cross_idx += 1
        if cross_idx % 200 == 0 or cross_idx == 1:
            print(f"  baseline {cross_idx}/{n_cross}  patches so far: {len(all_patches)}")

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

        all_patches.extend(patches)
        all_offsets.extend(offsets)
        baselines_used += 1

    if not all_patches:
        raise RuntimeError("no patches extracted — check flag fraction thresholds")

    patches_arr    = np.stack(all_patches, axis=0).astype(np.float32)
    offsets_arr    = np.array(all_offsets, dtype=np.int32)
    patch_freq_min = freqs[offsets_arr].astype(np.float32)
    patch_freq_max = freqs[np.minimum(offsets_arr + pf - 1, n_chan - 1)].astype(np.float32)

    print(f"column         : {col}")
    print(f"freq range     : {freq_min:.1f}-{freq_max:.1f} MHz  ({n_chan} channels)")
    print(f"baselines used : {baselines_used}  skipped: {baselines_skipped}  autocorr: {autocorr.sum()}")
    print(f"patches total  : {len(all_patches)}  shape: ({pt}, {pf})")

    if args.waterfall_out:
        wf_avg   = np.where(wf_count > 0, wf_sum / wf_count, 0.0).astype(np.float32)
        flag_avg = (wf_count == 0)
        np.save(args.waterfall_out + '.npy',       wf_avg)
        np.save(args.waterfall_out + '.meta.npy',  np.array([freq_min, freq_max]))
        np.save(args.waterfall_out + '.flags.npy', flag_avg)
        print(f"waterfall      -> {args.waterfall_out}.npy")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as hf:
        hf.create_dataset('clean',          data=patches_arr,    dtype=np.float32)
        hf.create_dataset('freq_min_patch', data=patch_freq_min, dtype=np.float32)
        hf.create_dataset('freq_max_patch', data=patch_freq_max, dtype=np.float32)
        hf.attrs['freq_min_mhz'] = freq_min
        hf.attrs['freq_max_mhz'] = freq_max
        hf.attrs['n_time']       = pt
        hf.attrs['n_freq']       = pf
        hf.attrs['n_patches']    = len(all_patches)

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
