import argparse
import numpy as np
import h5py
from pathlib import Path
from casacore.tables import table
from rfi_bands import LBAND_PERSISTENT_MHZ


def sigma_clip_flags(waterfall, flagged, sigma=3.0, n_iter=3):
    out = flagged.copy()
    n_time = waterfall.shape[0]
    for t in range(n_time):
        row = waterfall[t].copy().astype(np.float64)
        mask = out[t].copy()
        for _ in range(n_iter):
            valid = ~mask
            if valid.sum() < 4:
                break
            med = np.median(row[valid])
            mad = np.median(np.abs(row[valid] - med))
            thresh = med + sigma * mad * 1.4826
            new_flags = row > thresh
            if not np.any(new_flags & ~mask):
                break
            mask |= new_flags
        out[t] = mask
    return out


def divisive_norm(waterfall, flagged, smooth_bins=64):
    norm = waterfall.copy()
    n_time, n_chan = waterfall.shape
    for t in range(n_time):
        row = waterfall[t].copy()
        row[flagged[t]] = np.nan
        kernel = np.ones(smooth_bins) / smooth_bins
        valid = ~np.isnan(row)
        if valid.sum() < smooth_bins:
            continue
        smoothed = np.full(n_chan, np.nan)
        smoothed[valid] = np.convolve(row[valid], kernel, mode='same')
        nans = np.isnan(smoothed)
        if nans.all():
            continue
        idx = np.arange(n_chan)
        smoothed[nans] = np.interp(idx[nans], idx[~nans], smoothed[~nans])
        smoothed = np.clip(smoothed, 1e-6, None)
        norm[t] = waterfall[t] / smoothed
    return norm


def extract_patches(waterfall, flagged, pt, pf, st, sf, max_flag_frac, max_patches):
    n_time, n_chan = waterfall.shape
    patches, flag_patches, freq_offsets = [], [], []
    t = 0
    while t + pt <= n_time:
        f = 0
        while f + pf <= n_chan:
            pf_block = flagged[t:t+pt, f:f+pf]
            if pf_block.mean() <= max_flag_frac:
                patches.append(waterfall[t:t+pt, f:f+pf])
                flag_patches.append(pf_block)
                freq_offsets.append(f)
            f += sf
            if len(patches) >= max_patches:
                return patches, flag_patches, freq_offsets
        t += st
    return patches, flag_patches, freq_offsets


def main(args):
    ms = table(args.ms, readonly=True)
    query = []
    if args.field is not None:
        query.append(f"FIELD_ID == {args.field}")
    if args.max_time is not None or args.time_start > 0:
        all_times = np.unique(ms.getcol('TIME'))
        lo = min(args.time_start, len(all_times) - 1)
        hi = len(all_times) if args.max_time is None else min(lo + args.max_time, len(all_times))
        query.append(f"TIME >= {all_times[lo]}")
        query.append(f"TIME <= {all_times[hi - 1]}")
    if query:
        ms = ms.query(" AND ".join(query))

    cols = ms.colnames()
    if args.column is not None:
        col = args.column
    else:
        col = 'DATA'
        if 'CORRECTED_DATA' in cols:
            try:
                ms.getcell('CORRECTED_DATA', 0)
                col = 'CORRECTED_DATA'
            except Exception:
                pass

    data    = ms.getcol(col)
    flags   = ms.getcol('FLAG')
    times   = ms.getcol('TIME')
    ant1    = ms.getcol('ANTENNA1')
    ant2    = ms.getcol('ANTENNA2')
    ms.close()

    freqs_table = table(args.ms + '/SPECTRAL_WINDOW')
    freqs = freqs_table.getcol('CHAN_FREQ')[0] / 1e6
    freqs_table.close()

    chan_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    data  = data[:, chan_mask, :]
    flags = flags[:, chan_mask, :]
    freqs = freqs[chan_mask]
    n_chan = int(chan_mask.sum())

    for flo, fhi in LBAND_PERSISTENT_MHZ:
        flags[:, (freqs >= flo) & (freqs <= fhi), :] = True

    amp     = np.abs(data).mean(axis=2).astype(np.float32)
    flagged = flags.any(axis=2)

    unique_times = np.unique(times)
    n_time       = len(unique_times)
    n_row        = amp.shape[0]
    n_baseline   = n_row // n_time

    amp     = amp[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    flagged = flagged[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    ant1    = ant1[:n_baseline]
    ant2    = ant2[:n_baseline]

    autocorr = ant1 == ant2

    freq_min = freqs[0]
    freq_max = freqs[-1]
    pt, pf   = args.patch_time, args.patch_freq
    st, sf   = args.stride_time, args.stride_freq

    all_patches, all_raw, all_flags, all_freq_offsets = [], [], [], []
    baselines_used = 0
    baselines_skipped = 0

    for bl in range(n_baseline):
        if autocorr[bl]:
            continue

        wf = amp[:, bl, :]
        fm = flagged[:, bl, :]

        if fm.mean() > args.max_bl_flag_frac:
            baselines_skipped += 1
            continue

        if args.sigma_clip > 0:
            fm = sigma_clip_flags(wf, fm, sigma=args.sigma_clip)

        wf_norm = divisive_norm(wf, fm, smooth_bins=args.smooth_bins)

        patches, flag_patches, freq_offsets = extract_patches(
            wf_norm, fm, pt, pf, st, sf,
            args.max_flag_frac, args.max_patches_per_bl
        )

        if not patches:
            baselines_skipped += 1
            continue

        raw_patches, _, _ = extract_patches(
            wf, fm, pt, pf, st, sf,
            args.max_flag_frac, args.max_patches_per_bl
        )

        all_patches.extend(patches)
        all_raw.extend(raw_patches)
        all_flags.extend(flag_patches)
        all_freq_offsets.extend(freq_offsets)
        baselines_used += 1

    if not all_patches:
        raise RuntimeError("no patches extracted — check flag fraction thresholds")

    patches_arr     = np.stack(all_patches, axis=0).astype(np.float32)
    raw_arr         = np.stack(all_raw,     axis=0).astype(np.float32)
    flags_arr       = np.stack(all_flags,   axis=0).astype(np.float32)
    offsets_arr     = np.array(all_freq_offsets, dtype=np.int32)
    patch_freq_min  = freqs[offsets_arr].astype(np.float32)
    patch_freq_max  = freqs[np.minimum(offsets_arr + pf - 1, len(freqs) - 1)].astype(np.float32)

    print(f"column         : {col}")
    print(f"freq range     : {freq_min:.1f}–{freq_max:.1f} MHz  ({n_chan} channels)")
    print(f"baselines used : {baselines_used}  skipped: {baselines_skipped}  autocorr: {autocorr.sum()}")
    print(f"patches total  : {len(all_patches)}  shape: ({pt}, {pf})")
    print(f"mean flag frac : {flags_arr.mean():.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as hf:
        hf.create_dataset('data',           data=patches_arr,    dtype=np.float32)
        hf.create_dataset('data_raw',       data=raw_arr,        dtype=np.float32)
        hf.create_dataset('flags',          data=flags_arr,      dtype=np.float32)
        hf.create_dataset('freq_min_patch', data=patch_freq_min, dtype=np.float32)
        hf.create_dataset('freq_max_patch', data=patch_freq_max, dtype=np.float32)
        hf.attrs['freq_min_mhz']    = freq_min
        hf.attrs['freq_max_mhz']    = freq_max
        hf.attrs['n_time']          = pt
        hf.attrs['n_freq']          = pf
        hf.attrs['n_patches']       = len(all_patches)
        hf.attrs['baselines_used']  = baselines_used
        hf.attrs['smooth_bins']     = args.smooth_bins

    print(f"saved -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms',                  required=True)
    parser.add_argument('--output',              required=True)
    parser.add_argument('--column',              default=None)
    parser.add_argument('--field',               type=int,   default=None)
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
    parser.add_argument('--max-time',            type=int,   default=None)
    parser.add_argument('--time-start',          type=int,   default=0)
    parser.add_argument('--sigma-clip',          type=float, default=3.0)
    main(parser.parse_args())
