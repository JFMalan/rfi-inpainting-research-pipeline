import argparse
import numpy as np
import h5py
from pathlib import Path


FREQ_MIN_MHZ = 880.0
FREQ_MAX_MHZ = 1680.0

CLEAN_BANDS = [
    (880.0,  930.0),
    (960.0,  1163.0),
    (1299.0, 1524.0),
    (1630.0, 1680.0),
]

PERSISTENT_RFI = [
    (930.0,  960.0,  []),
    (1163.0, 1299.0, [1176.45, 1207.14, 1227.60, 1246.0]),
    (1530.0, 1630.0, [1575.42, 1602.0]),
]


def _clean_band_mask(freqs):
    mask = np.zeros(len(freqs), dtype=bool)
    for f_lo, f_hi in CLEAN_BANDS:
        mask |= (freqs >= f_lo) & (freqs <= f_hi)
    return mask


def corrupt(clean, rng, freqs):
    n_time, n_freq = clean.shape
    corrupted = clean.copy()
    mask = np.zeros((n_time, n_freq), dtype=np.float32)

    for f_lo, f_hi, line_centres in PERSISTENT_RFI:
        if rng.random() > 0.88:
            continue
        band = (freqs >= f_lo) & (freqs <= f_hi)
        n_band = band.sum()
        if n_band == 0:
            continue
        amplitude = rng.uniform(10.0, 50.0)
        rfi = amplitude * 0.3 * rng.standard_exponential((n_time, n_band)).astype(np.float32)
        band_freqs = freqs[band]
        for centre in line_centres:
            if not (f_lo <= centre <= f_hi):
                continue
            width = rng.uniform(3.0, 8.0)
            line_amp = amplitude * rng.uniform(0.5, 1.5)
            profile = np.exp(-0.5 * ((band_freqs - centre) / width) ** 2).astype(np.float32)
            rfi += line_amp * profile[np.newaxis, :]
        duty = rng.uniform(0.70, 1.0)
        time_on = rng.random(n_time) < duty
        rfi[~time_on] = 0.0
        corrupted[:, band] += rfi
        mask[np.ix_(time_on, band)] = 1.0

    clean_indices = np.where(_clean_band_mask(freqs))[0]
    if len(clean_indices) > 0:
        for _ in range(rng.integers(1, 8)):
            ch = clean_indices[rng.integers(0, len(clean_indices))]
            width = rng.integers(1, 4)
            ch_lo = max(0, ch - width)
            ch_hi = min(n_freq, ch + width + 1)
            amplitude = rng.uniform(3.0, 15.0)
            time_on = rng.random(n_time) < rng.uniform(0.02, 0.20)
            corrupted[np.ix_(time_on, np.arange(ch_lo, ch_hi))] += amplitude
            mask[np.ix_(time_on, np.arange(ch_lo, ch_hi))] = 1.0

    for _ in range(rng.integers(0, 3)):
        t0 = rng.integers(0, n_time)
        tw = rng.integers(2, max(3, n_time // 10))
        c0 = rng.integers(0, n_freq)
        cw = rng.integers(5, max(6, n_freq // 8))
        corrupted[t0:t0+tw, c0:c0+cw] += rng.uniform(5.0, 20.0)
        mask[t0:t0+tw, c0:c0+cw] = 1.0

    impulse = rng.random((n_time, n_freq)) < 0.0001
    corrupted[impulse] += rng.uniform(10.0, 50.0, size=impulse.sum()).astype(np.float32)
    mask[impulse] = 1.0

    return corrupted, mask


def main(args):
    rng = np.random.default_rng(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, 'r') as f:
        clean_patches = f['clean'][:]
        freq_min = f.attrs['freq_min_mhz']
        freq_max = f.attrs['freq_max_mhz']
        n_time = f.attrs['n_time']
        n_freq = f.attrs['n_freq']

    freqs = np.linspace(freq_min, freq_max, n_freq)
    n_patches = len(clean_patches)
    print(f"injecting RFI into {n_patches} patches")

    with h5py.File(out_path, 'w') as f:
        clean_ds = f.create_dataset('clean', data=clean_patches, dtype=np.float32)
        corrupted_ds = f.create_dataset('corrupted', shape=clean_patches.shape, dtype=np.float32)
        mask_ds = f.create_dataset('mask', shape=clean_patches.shape, dtype=np.float32)

        f.attrs['freq_min_mhz'] = freq_min
        f.attrs['freq_max_mhz'] = freq_max
        f.attrs['n_time'] = n_time
        f.attrs['n_freq'] = n_freq
        f.attrs['seed'] = args.seed

        for i, patch in enumerate(clean_patches):
            corrupted, mask = corrupt(patch, rng, freqs)
            corrupted_ds[i] = corrupted
            mask_ds[i] = mask
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{n_patches}")

    print(f"saved -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args())
