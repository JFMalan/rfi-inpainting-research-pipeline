import argparse
import sys
import numpy as np
import h5py
from pathlib import Path

from rfi_toolbox.data_generation.synthetic_generator import SyntheticDataGenerator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real'))
from rfi_bands import LBAND_PERSISTENT_MHZ


RFI_SCALE_MIN = 5.0    # RFI peaks at least 5x clean std
RFI_SCALE_MAX = 50.0   # RFI peaks at most 50x clean std


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


def persistent_bands(n_freq, freq_min, freq_max):
    f = np.linspace(freq_min, freq_max, n_freq)
    bands = []
    for lo, hi in LBAND_PERSISTENT_MHZ:
        idx = np.where((f >= lo) & (f <= hi))[0]
        if idx.size:
            bands.append((int(idx[0]), int(idx[-1] + 1)))
    return bands


def band_overlay(mask, rfi, clean_std, bands, target_frac, scale_min, scale_max,
                 persist_frac):
    n_time, n_freq = mask.shape

    def fill(t0, t1, f0, f1):
        mask[t0:t1, f0:f1] = 1.0
        if clean_std > 0:
            peak = np.random.uniform(scale_min, scale_max) * clean_std
            rfi[t0:t1, f0:f1] += peak * np.random.uniform(0.4, 1.0, size=(t1 - t0, f1 - f0))

    for f0, f1 in bands:
        if np.random.rand() > persist_frac:
            continue
        # real persistent bands are not solid: leave time gaps and partial freq coverage
        if np.random.rand() < 0.5:
            fill(0, n_time, f0, f1)
        else:
            t = 0
            while t < n_time:
                on = np.random.randint(int(n_time * 0.1), int(n_time * 0.5) + 1)
                fill(t, min(t + on, n_time), f0, f1)
                t += on + np.random.randint(int(n_time * 0.05), int(n_time * 0.3) + 1)

    guard = 0
    while mask.mean() < target_frac and guard < 300:
        guard += 1
        w = np.random.randint(2, 18)
        f0 = np.random.randint(0, max(1, n_freq - w))
        fill(0, n_time, f0, f0 + w)


def inject(clean_patch, gen, synth_cfg, bands, target_frac, scale_min, scale_max,
           persist_frac):
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
    rfi_amp = np.abs(waterfall[0, 0]).astype(np.float32).T   # (n_time, n_freq)
    rfi_mask = mask[0, 0].astype(np.float32).T

    clean_std = clean_patch.std()
    if clean_std > 0 and rfi_amp.max() > 0:
        target_peak = np.random.uniform(scale_min, scale_max) * clean_std
        rfi_amp = rfi_amp * (target_peak / rfi_amp.max())

    band_overlay(rfi_mask, rfi_amp, clean_std, bands, target_frac,
                 scale_min, scale_max, persist_frac)

    corrupted = clean_patch + rfi_amp
    return corrupted.astype(np.float32), rfi_mask


def main(args):
    np.random.seed(args.seed)

    fin = h5py.File(args.input, 'r')
    n_time = int(fin.attrs['n_time'])
    n_freq = int(fin.attrs['n_freq'])
    freq_min = float(fin.attrs['freq_min_mhz'])
    freq_max = float(fin.attrs['freq_max_mhz'])
    attrs = dict(fin.attrs)
    clean_ds_in = fin['clean']
    n_patches = clean_ds_in.shape[0]
    other_keys = [k for k in fin if k != 'clean']

    synth_cfg = _synth_config(n_freq, n_time)
    gen = SyntheticDataGenerator({"synthetic": synth_cfg})
    bands = persistent_bands(n_freq, freq_min, freq_max)
    print(f"injecting RFI into {n_patches} baselines ({n_time}x{n_freq}), "
          f"{len(bands)} persistent bands, target frac {args.target_frac}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk = 256
    fracs = []
    with h5py.File(out_path, 'w') as f:
        clean_out    = f.create_dataset('clean',     shape=clean_ds_in.shape, dtype=np.float32)
        corrupted_ds = f.create_dataset('corrupted', shape=clean_ds_in.shape, dtype=np.float32)
        mask_ds      = f.create_dataset('mask',      shape=clean_ds_in.shape, dtype=np.float32)
        for k in other_keys:
            src = fin[k]
            dst = f.create_dataset(k, shape=src.shape, dtype=src.dtype)
            for s in range(0, src.shape[0], chunk):
                e = min(s + chunk, src.shape[0])
                dst[s:e] = src[s:e]
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs['seed'] = args.seed

        for s in range(0, n_patches, chunk):
            e = min(s + chunk, n_patches)
            block = clean_ds_in[s:e]
            clean_out[s:e] = block
            for j, patch in enumerate(block):
                corrupted, mask = inject(patch, gen, synth_cfg, bands,
                                         args.target_frac, args.scale_min,
                                         args.scale_max, args.persist_frac)
                corrupted_ds[s + j] = corrupted
                mask_ds[s + j] = mask
                fracs.append(float(mask.mean()))
            print(f"  {e}/{n_patches}  mean flag frac {np.mean(fracs):.3f}", flush=True)

    fin.close()
    print(f"mean flag frac : {np.mean(fracs):.3f}  (target {args.target_frac})", flush=True)
    print(f"saved -> {out_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--target-frac',  type=float, default=0.40)
    parser.add_argument('--persist-frac', type=float, default=0.7)
    parser.add_argument('--scale-min',    type=float, default=RFI_SCALE_MIN)
    parser.add_argument('--scale-max',    type=float, default=RFI_SCALE_MAX)
    main(parser.parse_args())
