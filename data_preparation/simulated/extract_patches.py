import argparse
import numpy as np
import h5py
from pathlib import Path


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


def main(args):
    waterfall = np.load(args.waterfall)
    freq_min, freq_max = np.load(args.waterfall.replace('.npy', '.meta.npy'))

    flags_path = args.waterfall.replace('.npy', '.flags.npy')
    flag_map = np.load(flags_path) if Path(flags_path).exists() else np.zeros(waterfall.shape, dtype=bool)

    freqs = np.linspace(freq_min, freq_max, waterfall.shape[1])

    waterfall = divisive_norm(waterfall, flag_map.astype(bool), smooth_bins=args.smooth_bins)

    n_time, n_chan = waterfall.shape
    pt, pf = args.patch_time, args.patch_freq
    st, sf = args.stride_time, args.stride_freq

    patches, freq_offsets = [], []
    skipped = 0
    t = 0
    while t + pt <= n_time and len(patches) < args.max_patches:
        f = 0
        while f + pf <= n_chan and len(patches) < args.max_patches:
            patch_flags = flag_map[t:t+pt, f:f+pf]
            if patch_flags.mean() > args.max_flag_fraction:
                skipped += 1
                f += sf
                continue
            patches.append(waterfall[t:t+pt, f:f+pf])
            freq_offsets.append(f)
            f += sf
        t += st

    if not patches:
        raise RuntimeError("no patches passed the flag fraction threshold")

    patches_arr    = np.stack(patches, axis=0).astype(np.float32)
    offsets_arr    = np.array(freq_offsets, dtype=np.int32)
    patch_freq_min = freqs[offsets_arr].astype(np.float32)
    patch_freq_max = freqs[np.minimum(offsets_arr + pf - 1, len(freqs) - 1)].astype(np.float32)

    print(f"extracted {len(patches)} patches ({pt}x{pf}), skipped {skipped} (>{args.max_flag_fraction*100:.0f}% flagged)")

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
        hf.attrs['n_patches']    = len(patches)

    print(f"saved -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--waterfall',          required=True)
    parser.add_argument('--output',             required=True)
    parser.add_argument('--patch-time',         type=int,   default=256)
    parser.add_argument('--patch-freq',         type=int,   default=256)
    parser.add_argument('--stride-time',        type=int,   default=64)
    parser.add_argument('--stride-freq',        type=int,   default=64)
    parser.add_argument('--max-patches',        type=int,   default=500)
    parser.add_argument('--max-flag-fraction',  type=float, default=0.5)
    parser.add_argument('--smooth-bins',        type=int,   default=64)
    main(parser.parse_args())
