import argparse
import sys
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real'))
from rfi_bands import LBAND_PERSISTENT_MHZ


RFI_SCALE_MIN = 5.0
RFI_SCALE_MAX = 50.0


def persistent_cols(n_freq, freq_min, freq_max):
    cols = np.zeros(n_freq, dtype=bool)
    f = np.linspace(freq_min, freq_max, n_freq)
    for lo, hi in LBAND_PERSISTENT_MHZ:
        cols |= (f >= lo) & (f <= hi)
    return cols


def inject(clean, persist_cols, target_frac, scale_min, scale_max, persist_frac=0.7):
    n_time, n_freq = clean.shape
    mask = np.zeros((n_time, n_freq), dtype=np.float32)
    rfi = np.zeros((n_time, n_freq), dtype=np.float32)
    clean_std = clean.std()

    def add_rfi(sl_t, sl_f, area_shape):
        mask[sl_t, sl_f] = 1.0
        if clean_std > 0:
            peak = np.random.uniform(scale_min, scale_max) * clean_std
            rfi[sl_t, sl_f] += peak * np.random.uniform(0.4, 1.0, size=area_shape)

    if np.random.rand() < persist_frac:
        mask[:, persist_cols] = 1.0

    n_burst = np.random.randint(2, 6)
    for _ in range(n_burst):
        w = np.random.randint(2, 8)
        t0 = np.random.randint(0, max(1, n_time - w))
        add_rfi(slice(t0, t0 + w), slice(None), (w, n_freq))

    guard = 0
    while mask.mean() < target_frac and guard < 300:
        guard += 1
        w = np.random.randint(2, 18)
        f0 = np.random.randint(0, max(1, n_freq - w))
        add_rfi(slice(None), slice(f0, f0 + w), (n_time, w))

    corrupted = clean + rfi
    return corrupted.astype(np.float32), mask


def main(args):
    np.random.seed(args.seed)

    fin = h5py.File(args.input, 'r')
    n_freq = int(fin.attrs['n_freq'])
    n_time = int(fin.attrs['n_time'])
    freq_min = float(fin.attrs['freq_min_mhz'])
    freq_max = float(fin.attrs['freq_max_mhz'])
    attrs = dict(fin.attrs)
    clean_in = fin['clean']
    n = clean_in.shape[0]
    other_keys = [k for k in fin if k != 'clean']

    persist = persistent_cols(n_freq, freq_min, freq_max)
    print(f"injecting RFI into {n} baselines ({n_time}x{n_freq}), "
          f"persistent cols {persist.sum()}/{n_freq}, target frac {args.target_frac}",
          flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk = 256
    fracs = []
    with h5py.File(out_path, 'w') as f:
        clean_out = f.create_dataset('clean',     shape=clean_in.shape, dtype=np.float32)
        corr_ds   = f.create_dataset('corrupted', shape=clean_in.shape, dtype=np.float32)
        mask_ds   = f.create_dataset('mask',      shape=clean_in.shape, dtype=np.float32)
        for k in other_keys:
            src = fin[k]
            dst = f.create_dataset(k, shape=src.shape, dtype=src.dtype)
            for s in range(0, src.shape[0], chunk):
                e = min(s + chunk, src.shape[0])
                dst[s:e] = src[s:e]
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs['seed'] = args.seed

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            block = clean_in[s:e]
            clean_out[s:e] = block
            for j, patch in enumerate(block):
                corrupted, mask = inject(patch, persist, args.target_frac,
                                         args.scale_min, args.scale_max,
                                         persist_frac=args.persist_frac)
                corr_ds[s + j] = corrupted
                mask_ds[s + j] = mask
                fracs.append(float(mask.mean()))
            print(f"  {e}/{n}  mean flag frac {np.mean(fracs):.3f}", flush=True)

    fin.close()
    print(f"mean flag frac : {np.mean(fracs):.3f}  (target {args.target_frac})", flush=True)
    print(f"saved -> {out_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--target-frac', type=float, default=0.40)
    parser.add_argument('--persist-frac', type=float, default=0.7)
    parser.add_argument('--scale-min',   type=float, default=RFI_SCALE_MIN)
    parser.add_argument('--scale-max',   type=float, default=RFI_SCALE_MAX)
    main(parser.parse_args())
