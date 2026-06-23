import argparse
import glob

import numpy as np
import h5py


def main(args):
    files = sorted(glob.glob(args.inputs))
    if not files:
        raise SystemExit(f"no files match {args.inputs}")
    print(f"merging {len(files)} window files")
    keys = ['data', 'data_raw', 'flags', 'freq_min_patch', 'freq_max_patch']
    acc = {k: [] for k in keys}
    win = []
    for wi, f in enumerate(files):
        with h5py.File(f, 'r') as hf:
            n = hf['data'].shape[0]
            for k in keys:
                acc[k].append(hf[k][:])
            win.append(np.full(n, wi, dtype=np.int32))
        print(f"  {f}: {n} patches")
    merged = {k: np.concatenate(acc[k], axis=0) for k in keys}
    window_id = np.concatenate(win)
    with h5py.File(args.output, 'w') as hf:
        for k, v in merged.items():
            hf.create_dataset(k, data=v)
        hf.create_dataset('window_id', data=window_id)
        hf.attrs['n_patches'] = merged['data'].shape[0]
        hf.attrs['n_windows'] = len(files)
    print(f"total {merged['data'].shape[0]} patches across {len(files)} windows -> {args.output}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--output', required=True)
    main(ap.parse_args())
