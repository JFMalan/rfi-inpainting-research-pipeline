import argparse
import numpy as np
import h5py
from pathlib import Path
from casacore.tables import table


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
        # fill edges and flagged gaps with nearest valid smoothed value
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
    patches, flag_patches = [], []
    t = 0
    while t + pt <= n_time:
        f = 0
        while f + pf <= n_chan:
            pf_block = flagged[t:t+pt, f:f+pf]
            if pf_block.mean() <= max_flag_frac:
                patches.append(waterfall[t:t+pt, f:f+pf])
                flag_patches.append(pf_block)
            f += sf
            if len(patches) >= max_patches:
                return patches, flag_patches
        t += st
    return patches, flag_patches


def main(args):
    ms = table(args.ms, readonly=True)
    if args.field is not None:
        ms = ms.query(f"FIELD_ID == {args.field}")

    cols = ms.colnames()
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
    for lo, hi in args.exclude_bands:
        chan_mask &= ~((freqs >= lo) & (freqs <= hi))
    data  = data[:, chan_mask, :]
    flags = flags[:, chan_mask, :]
    freqs = freqs[chan_mask]
    n_chan = int(chan_mask.sum())

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

    all_patches, all_flags = [], []
    baselines_used = 0
    baselines_skipped = 0

    for bl in range(n_baseline):
        if autocorr[bl]:
            continue

        wf = amp[:, bl, :]       # (n_time, n_chan)
        fm = flagged[:, bl, :]   # (n_time, n_chan)

        if fm.mean() > args.max_bl_flag_frac:
            baselines_skipped += 1
            continue

        wf_norm = divisive_norm(wf, fm, smooth_bins=args.smooth_bins)

        patches, flag_patches = extract_patches(
            wf_norm, fm, pt, pf, st, sf,
            args.max_flag_frac, args.max_patches_per_bl
        )

        if not patches:
            baselines_skipped += 1
            continue

        all_patches.extend(patches)
        all_flags.extend(flag_patches)
        baselines_used += 1

    if not all_patches:
        raise RuntimeError("no patches extracted — check flag fraction thresholds")

    patches_arr = np.stack(all_patches, axis=0).astype(np.float32)
    flags_arr   = np.stack(all_flags,   axis=0).astype(np.float32)

    print(f"column         : {col}")
    print(f"freq range     : {freq_min:.1f}–{freq_max:.1f} MHz  ({n_chan} channels)")
    print(f"baselines used : {baselines_used}  skipped: {baselines_skipped}  autocorr: {autocorr.sum()}")
    print(f"patches total  : {len(all_patches)}  shape: ({pt}, {pf})")
    print(f"mean flag frac : {flags_arr.mean():.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as hf:
        hf.create_dataset('data',  data=patches_arr, dtype=np.float32)
        hf.create_dataset('flags', data=flags_arr,   dtype=np.float32)
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
    parser.add_argument('--exclude-bands',       type=float, nargs=2, action='append',
                        default=[[1170, 1300], [1525, 1630]],
                        metavar=('LO', 'HI'))
    main(parser.parse_args())
