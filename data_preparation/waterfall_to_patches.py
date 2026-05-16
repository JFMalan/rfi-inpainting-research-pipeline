"""
Step 2 of extraction: slices waterfall numpy array into 256x256 patches and saves to HDF5.
Run in ASTRO-PY3.10 container (has h5py, no casacore).
"""
import argparse
import numpy as np
import h5py
from pathlib import Path


def main(args):
    waterfall = np.load(args.waterfall)
    freq_min, freq_max = np.load(args.waterfall.replace('.npy', '.meta.npy'))

    n_time, n_chan = waterfall.shape
    pt, pf = args.patch_time, args.patch_freq
    st, sf = args.stride_time, args.stride_freq

    patches = []
    t = 0
    while t + pt <= n_time and len(patches) < args.max_patches:
        f = 0
        while f + pf <= n_chan and len(patches) < args.max_patches:
            patches.append(waterfall[t:t+pt, f:f+pf])
            f += sf
        t += st

    patches = np.stack(patches, axis=0)
    print(f"extracted {len(patches)} patches ({pt}x{pf})")

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
    main(parser.parse_args())
