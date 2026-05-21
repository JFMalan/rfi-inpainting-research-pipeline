import argparse
import numpy as np
import h5py
from pathlib import Path

from rfi_toolbox.data_generation.synthetic_generator import SyntheticDataGenerator


def _synth_config(n_freq, n_time):
    return {
        "num_channels": n_freq,
        "num_times": n_time,
        "noise_mjy": 0.0,
        "rfi_power_min": 60.0,
        "rfi_power_max": 1500.0,
        "enable_bandpass_rolloff": False,
        "bandpass_polynomial_order": 8,
        "num_polarizations": 1,
        "polarization_correlation": 0.8,
        "rfi_types": [
            "narrowband_persistent",
            "broadband_persistent",
            "narrowband_intermittent",
            "narrowband_bursty",
            "broadband_bursty",
            "frequency_sweep",
        ],
        "rfi_type_counts": {
            "narrowband_persistent": [1, 2],
            "broadband_persistent": [0, 1],
            "narrowband_intermittent": [0, 1],
            "narrowband_bursty": [0, 1],
            "broadband_bursty": [0, 1],
            "frequency_sweep": [0, 1],
        },
    }


RFI_SCALE_MIN = 5.0    # RFI peaks at least 5× clean std — clearly visible
RFI_SCALE_MAX = 50.0   # RFI peaks at most 50× clean std


def inject(clean_patch, gen, synth_cfg):
    n_time, n_freq = clean_patch.shape
    rfi_config = gen._parse_rfi_config(synth_cfg)

    waterfall, mask, _ = gen._generate_single_sample(
        num_channels=n_freq,
        num_times=n_time,
        noise_level=0.0,
        rfi_power_min=synth_cfg["rfi_power_min"],
        rfi_power_max=synth_cfg["rfi_power_max"],
        rfi_config=rfi_config,
        enable_bandpass=False,
        bandpass_order=8,
        num_polarizations=1,
        pol_corr=0.8,
        synth_config=synth_cfg,
    )
    # waterfall: (1, 1, n_freq, n_time) complex64
    rfi_amp = np.abs(waterfall[0, 0]).astype(np.float32)  # (n_freq, n_time)
    rfi_mask = mask[0, 0].astype(np.float32)               # (n_freq, n_time)

    # rfi_toolbox internal units are unrelated to the MS amplitude scale,
    # so rescale to a random multiple of the clean patch std
    clean_std = clean_patch.std()
    if clean_std > 0 and rfi_amp.max() > 0:
        target_peak = np.random.uniform(RFI_SCALE_MIN, RFI_SCALE_MAX) * clean_std
        rfi_amp = rfi_amp * (target_peak / rfi_amp.max())

    corrupted = clean_patch + rfi_amp.T  # patches are (n_time, n_freq)
    return corrupted, rfi_mask.T


def main(args):
    np.random.seed(args.seed)

    with h5py.File(args.input, 'r') as f:
        clean_patches = f['clean'][:]
        freq_min = f.attrs['freq_min_mhz']
        freq_max = f.attrs['freq_max_mhz']
        n_time = int(f.attrs['n_time'])
        n_freq = int(f.attrs['n_freq'])

    synth_cfg = _synth_config(n_freq, n_time)
    gen = SyntheticDataGenerator({"synthetic": synth_cfg})

    n_patches = len(clean_patches)
    print(f"injecting RFI into {n_patches} patches")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('clean', data=clean_patches, dtype=np.float32)
        corrupted_ds = f.create_dataset('corrupted', shape=clean_patches.shape, dtype=np.float32)
        mask_ds = f.create_dataset('mask', shape=clean_patches.shape, dtype=np.float32)

        f.attrs['freq_min_mhz'] = freq_min
        f.attrs['freq_max_mhz'] = freq_max
        f.attrs['n_time'] = n_time
        f.attrs['n_freq'] = n_freq
        f.attrs['seed'] = args.seed

        for i, patch in enumerate(clean_patches):
            corrupted, mask = inject(patch, gen, synth_cfg)
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
