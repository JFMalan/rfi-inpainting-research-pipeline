import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter


FREQ_MIN_MHZ = 880.0
FREQ_MAX_MHZ = 1680.0

# Frank et al. (2021) Table 1 — usable L-band bands
CLEAN_BANDS = [
    (880.0,  930.0),   # Band A
    (960.0,  1163.0),  # Band B
    (1299.0, 1524.0),  # Band C
    (1630.0, 1680.0),  # Band D
]

# Persistent RFI regions with known spectral line centres (MHz) and widths (MHz).
# Sources: Frank et al. (2021), TABASCAL (Finlay et al. 2025), kathprfi occupancy stats.
# Each entry: (band_lo, band_hi, [(line_centre, line_width_mhz), ...])
# Lines are GNSS carriers: GPS L1=1575.42, L2=1227.60, L5=1176.45;
# GLONASS L1≈1602, L2≈1246; Galileo E1=1575.42, E5b=1207.14
PERSISTENT_RFI = [
    # Mobile/aeronautical band — no discrete lines, wideband occupancy
    (930.0,  960.0,  []),
    # GNSS L2/L5 band — GPS L2 1227.6, GPS L5 1176.45, GLONASS L2 ~1246, Galileo E5b 1207.14
    (1163.0, 1299.0, [1176.45, 1207.14, 1227.60, 1246.0]),
    # GNSS L1 band — GPS/Galileo L1 1575.42, GLONASS L1 ~1602
    (1530.0, 1630.0, [1575.42, 1602.0]),
]


def clean_spectrogram(rng, n_time, n_freq):
    freqs = np.linspace(FREQ_MIN_MHZ, FREQ_MAX_MHZ, n_freq, dtype=np.float32)

    # Rayleigh-distributed amplitude noise — correct statistics for averaged
    # visibility amplitudes (Rice distribution → Rayleigh when SNR is low)
    sigma = rng.uniform(0.8, 1.2)
    noise = rng.rayleigh(scale=sigma, size=(n_time, n_freq)).astype(np.float32)

    # Smooth continuum emission with realistic power-law spectral index
    n_sources = rng.integers(1, 5)
    source = np.zeros((n_time, n_freq), dtype=np.float32)
    for _ in range(n_sources):
        s = gaussian_filter(
            rng.exponential(scale=rng.uniform(0.1, 0.5), size=(n_time, n_freq)).astype(np.float32),
            sigma=(rng.uniform(1.0, 4.0), rng.uniform(2.0, 6.0))
        )
        spectral_index = rng.uniform(0.5, 2.5)
        s *= (freqs / FREQ_MIN_MHZ)[np.newaxis, :] ** (-spectral_index)
        source += s

    return noise + source


def _band_channel_mask(freqs, f_lo, f_hi):
    return (freqs >= f_lo) & (freqs <= f_hi)


def _clean_band_mask(freqs):
    mask = np.zeros(len(freqs), dtype=bool)
    for f_lo, f_hi in CLEAN_BANDS:
        mask |= _band_channel_mask(freqs, f_lo, f_hi)
    return mask


def inject_rfi(clean, rng, n_time, n_freq):
    corrupted = clean.copy()
    mask = np.zeros((n_time, n_freq), dtype=np.float32)
    freqs = np.linspace(FREQ_MIN_MHZ, FREQ_MAX_MHZ, n_freq)

    # --- Persistent GNSS/satellite RFI ---
    # Present in ~85-90% of samples per band (kathprfi occupancy).
    # Modelled as Gaussian spectral profiles on known carrier frequencies
    # (Finlay et al. 2025) plus a wideband noise floor within the band.
    # Temporal duty cycle 70-100% — satellites are mostly up but not always
    # in beam. Amplitude 10-50x noise floor (strong satellite signal).
    for f_lo, f_hi, line_centres in PERSISTENT_RFI:
        if rng.random() > 0.88:
            continue

        band = _band_channel_mask(freqs, f_lo, f_hi)
        n_band = band.sum()
        if n_band == 0:
            continue

        amplitude = rng.uniform(10.0, 50.0)
        # Wideband noise floor within the band
        rfi = amplitude * 0.3 * rng.standard_exponential((n_time, n_band)).astype(np.float32)

        # Spectral lines at known GNSS carrier frequencies
        band_freqs = freqs[band]
        for centre_mhz in line_centres:
            if not (f_lo <= centre_mhz <= f_hi):
                continue
            line_width = rng.uniform(3.0, 8.0)  # MHz, per TABASCAL Gaussian profiles
            line_amp = amplitude * rng.uniform(0.5, 1.5)
            profile = np.exp(-0.5 * ((band_freqs - centre_mhz) / line_width) ** 2).astype(np.float32)
            rfi += line_amp * profile[np.newaxis, :]

        # Temporal structure: satellite in/out of beam
        duty = rng.uniform(0.70, 1.0)
        time_on = rng.random(n_time) < duty
        rfi[~time_on] = 0.0

        corrupted[:, band] += rfi
        mask[np.ix_(time_on, band)] = 1.0

    # --- Narrowband RFI in clean bands ---
    # Ground-based transmitters: intermittent, placed only in bands A-D.
    # Occupancy <5% per kathprfi. Persistent in frequency but intermittent
    # in time (2-20% duty cycle per line).
    clean_mask = _clean_band_mask(freqs)
    clean_indices = np.where(clean_mask)[0]
    if len(clean_indices) > 0:
        n_narrowband = rng.integers(1, 8)
        for _ in range(n_narrowband):
            ch = clean_indices[rng.integers(0, len(clean_indices))]
            width = rng.integers(1, 4)
            ch_lo = max(0, ch - width)
            ch_hi = min(n_freq, ch + width + 1)
            amplitude = rng.uniform(3.0, 15.0)
            # Intermittent: only present in some time bins
            duty = rng.uniform(0.02, 0.20)
            time_on = rng.random(n_time) < duty
            corrupted[np.ix_(time_on, np.arange(ch_lo, ch_hi))] += amplitude
            mask[np.ix_(time_on, np.arange(ch_lo, ch_hi))] = 1.0

    # --- Broadband RFI (aircraft, DME, radar) ---
    # Time-frequency blobs, rare (1-2 per patch), moderate amplitude.
    # Can land anywhere in the band including persistent RFI regions.
    n_broadband = rng.integers(0, 3)
    for _ in range(n_broadband):
        t_start = rng.integers(0, n_time)
        t_width = rng.integers(2, max(3, n_time // 10))
        ch_start = rng.integers(0, n_freq)
        ch_width = rng.integers(5, max(6, n_freq // 8))
        t_end = min(n_time, t_start + t_width)
        ch_end = min(n_freq, ch_start + ch_width)
        amplitude = rng.uniform(5.0, 20.0)
        corrupted[t_start:t_end, ch_start:ch_end] += amplitude
        mask[t_start:t_end, ch_start:ch_end] = 1.0

    # --- Impulse RFI ---
    # Rare transient spikes. ~0.0001 per pixel per van Zyl et al. (2024).
    impulse = rng.random((n_time, n_freq)) < 0.0001
    corrupted[impulse] += rng.uniform(10.0, 50.0, size=impulse.sum()).astype(np.float32)
    mask[impulse] = 1.0

    return corrupted, mask


def generate(args):
    rng = np.random.default_rng(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_time, n_freq, n_samples = args.n_time, args.n_freq, args.n_samples
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
