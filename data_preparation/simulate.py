import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter


FREQ_MIN_MHZ = 880.0
FREQ_MAX_MHZ = 1680.0

RFI_BANDS = [
    (933.0, 960.0),
    (1163.0, 1299.0),
    (1530.0, 1630.0),
]


def clean_spectrogram(rng, n_time, n_freq):
    noise = rng.standard_normal((n_time, n_freq)).astype(np.float32)
    noise = np.abs(noise)

    source = gaussian_filter(
        rng.exponential(scale=0.3, size=(n_time, n_freq)).astype(np.float32),
        sigma=(2.0, 4.0)
    )

    freqs = np.linspace(FREQ_MIN_MHZ, FREQ_MAX_MHZ, n_freq, dtype=np.float32)
    spectral_index = rng.uniform(0.5, 1.5)
    spectral_envelope = (freqs / FREQ_MIN_MHZ) ** (-spectral_index)
    source *= spectral_envelope[np.newaxis, :]

    return noise + source


def inject_rfi(clean, rng, n_time, n_freq):
    corrupted = clean.copy()
    mask = np.zeros((n_time, n_freq), dtype=np.float32)

    freqs = np.linspace(FREQ_MIN_MHZ, FREQ_MAX_MHZ, n_freq)

    persistent_mask = np.zeros(n_freq, dtype=bool)
    for f_lo, f_hi in RFI_BANDS:
        persistent_mask |= (freqs >= f_lo) & (freqs <= f_hi)

    if persistent_mask.any():
        amplitude = rng.uniform(5.0, 20.0)
        corrupted[:, persistent_mask] += amplitude * rng.standard_exponential(
            (n_time, persistent_mask.sum())
        ).astype(np.float32)
        mask[:, persistent_mask] = 1.0

    n_narrowband = rng.integers(2, 8)
    for _ in range(n_narrowband):
        ch = rng.integers(0, n_freq)
        width = rng.integers(1, 4)
        ch_lo = max(0, ch - width)
        ch_hi = min(n_freq, ch + width + 1)
        amplitude = rng.uniform(3.0, 15.0)
        corrupted[:, ch_lo:ch_hi] += amplitude
        mask[:, ch_lo:ch_hi] = 1.0

    n_broadband = rng.integers(1, 4)
    for _ in range(n_broadband):
        t_start = rng.integers(0, n_time)
        t_width = rng.integers(2, max(3, n_time // 8))
        ch_start = rng.integers(0, n_freq)
        ch_width = rng.integers(10, max(11, n_freq // 6))
        t_end = min(n_time, t_start + t_width)
        ch_end = min(n_freq, ch_start + ch_width)
        amplitude = rng.uniform(8.0, 30.0)
        corrupted[t_start:t_end, ch_start:ch_end] += amplitude
        mask[t_start:t_end, ch_start:ch_end] = 1.0

    impulse_mask = rng.random((n_time, n_freq)) < 0.0002
    corrupted[impulse_mask] += rng.uniform(10.0, 50.0, size=impulse_mask.sum()).astype(np.float32)
    mask[impulse_mask] = 1.0

    return corrupted, mask


def generate(args):
    rng = np.random.default_rng(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_time = args.n_time
    n_freq = args.n_freq
    n_samples = args.n_samples

    print(f"generating {n_samples} samples ({n_time}x{n_freq}) -> {out_path}")

    with h5py.File(out_path, "w") as f:
        clean_ds = f.create_dataset("clean", shape=(n_samples, n_time, n_freq), dtype=np.float32)
        corrupted_ds = f.create_dataset("corrupted", shape=(n_samples, n_time, n_freq), dtype=np.float32)
        mask_ds = f.create_dataset("mask", shape=(n_samples, n_time, n_freq), dtype=np.float32)

        f.attrs["freq_min_mhz"] = FREQ_MIN_MHZ
        f.attrs["freq_max_mhz"] = FREQ_MAX_MHZ
        f.attrs["n_time"] = n_time
        f.attrs["n_freq"] = n_freq
        f.attrs["seed"] = args.seed

        for i in range(n_samples):
            clean = clean_spectrogram(rng, n_time, n_freq)
            corrupted, mask = inject_rfi(clean, rng, n_time, n_freq)

            clean_ds[i] = clean
            corrupted_ds[i] = corrupted
            mask_ds[i] = mask

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n_samples}")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_time", type=int, default=256)
    parser.add_argument("--n_freq", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    generate(parser.parse_args())
