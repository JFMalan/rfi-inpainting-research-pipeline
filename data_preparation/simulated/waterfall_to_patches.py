import argparse
import numpy as np
import h5py
from pathlib import Path


def main(args):
    waterfall = np.load(args.waterfall)
    freq_min, freq_max = np.load(args.waterfall.replace('.npy', '.meta.npy'))

    flags_path = args.waterfall.replace('.npy', '.flags.npy')
    flag_map = np.load(flags_path) if Path(flags_path).exists() else np.zeros_like(waterfall)

    n_time, n_chan = waterfall.shape
    pt, pf = args.patch_time, args.patch_freq
    st, sf = args.stride_time, args.stride_freq

    patches = []
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
            f += sf
        t += st

    if not patches:
        raise RuntimeError("no patches passed the flag fraction threshold")

    patches = np.stack(patches, axis=0)
    print(f"extracted {len(patches)} patches ({pt}x{pf}), skipped {skipped} (>{args.max_flag_fraction*100:.0f}% flagged)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as hf:
        hf.create_dataset('clean', data=patches, dtype=np.float32)
        hf.attrs['freq_min_mhz'] = freq_min
        hf.attrs['freq_max_mhz'] = freq_max
        hf.attrs['n_time'] = pt
        hf.attrs['n_freq'] = pf
        hf.attrs['n_patches'] = len(patches)

    print(f"saved -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--waterfall', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--patch-time', type=int, default=256)
    parser.add_argument('--patch-freq', type=int, default=256)
    parser.add_argument('--stride-time', type=int, default=64)
    parser.add_argument('--stride-freq', type=int, default=64)
    parser.add_argument('--max-patches', type=int, default=500)
    parser.add_argument('--max-flag-fraction', type=float, default=0.5)
    main(parser.parse_args())
