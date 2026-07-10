import argparse
import sys
import time
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real'))
from rfi_bands import LBAND_PERSISTENT_MHZ
from tiling import freq_tile_starts, freq_tile_width, time_extent, time_window_starts


RFI_SCALE_MIN = 5.0
RFI_SCALE_MAX = 50.0


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
    rfi_amp = np.abs(waterfall[0, 0]).astype(np.float32).T
    rfi_mask = mask[0, 0].astype(np.float32).T

    clean_std = clean_patch.std()
    if clean_std > 0 and rfi_amp.max() > 0:
        target_peak = np.random.uniform(scale_min, scale_max) * clean_std
        rfi_amp = rfi_amp * (target_peak / rfi_amp.max())

    band_overlay(rfi_mask, rfi_amp, clean_std, bands, target_frac,
                 scale_min, scale_max, persist_frac)

    corrupted = clean_patch + rfi_amp
    return corrupted.astype(np.float32), rfi_mask


def controlled_spans(n_freq, band_width, target_frac):
    # deterministic full-time frequency stripes of fixed width, evenly spaced across the band.
    # n_bands set so total flagged fraction ~ target_frac, isolating WIDTH from amount-flagged.
    n_bands = max(1, int(round(target_frac * n_freq / band_width)))
    centers = np.linspace(0, n_freq, n_bands + 2)[1:-1]
    spans = []
    for c in centers:
        f0 = int(round(c - band_width / 2))
        f0 = max(0, min(n_freq - band_width, f0))
        spans.append((f0, f0 + band_width))
    return spans


def controlled_inject(clean_patch, spans):
    mask = np.zeros_like(clean_patch)
    for f0, f1 in spans:
        mask[:, f0:f1] = 1.0
    return clean_patch.astype(np.float32), mask.astype(np.float32)


def resize_hw(arr, h, w, order=1):
    if arr.shape == (h, w):
        return arr.astype(np.float32)
    from skimage.transform import resize
    return resize(arr, (h, w), order=order, mode='edge',
                  anti_aliasing=(order > 0), preserve_range=True).astype(np.float32)


def resize_phase_hw(ph, h, w):
    return np.arctan2(resize_hw(np.sin(ph), h, w), resize_hw(np.cos(ph), h, w))


def main(args):
    np.random.seed(args.seed)
    sz = args.img_size

    fin = h5py.File(args.input, 'r')
    n_cross = fin['clean'].shape[0]
    clean_target = 'amp_target' in fin
    full_n_time = int(fin.attrs['full_n_time'])
    full_n_chan = int(fin.attrs['full_n_chan'])
    freq_min = float(fin.attrs['freq_min_mhz'])
    freq_max = float(fin.attrs['freq_max_mhz'])
    chan_lo = int(fin.attrs['chan_lo'])
    freqs = np.linspace(freq_min, freq_max, full_n_chan)

    starts = freq_tile_starts(full_n_chan, sz)
    nc = freq_tile_width(full_n_chan, sz)
    t_starts = time_window_starts(full_n_time, sz)
    th = min(sz, full_n_time)
    n_tiles = len(starts)
    cap = n_cross * n_tiles * len(t_starts)

    controlled = args.band_width and args.band_width > 0
    if controlled:
        spans = controlled_spans(full_n_chan, args.band_width, args.target_frac)
        gen = synth_cfg = bands = None
        print(f"CONTROLLED mode: {len(spans)} bands of width {args.band_width} ch "
              f"(target frac {args.target_frac}); spans={spans}", flush=True)
    else:
        from rfi_toolbox.data_generation.synthetic_generator import SyntheticDataGenerator
        synth_cfg = _synth_config(full_n_chan, full_n_time)
        gen = SyntheticDataGenerator({"synthetic": synth_cfg})
        bands = persistent_bands(full_n_chan, freq_min, freq_max)
        print(f"native {full_n_time}x{full_n_chan}, {len(bands)} persistent bands, "
              f"target frac {args.target_frac}", flush=True)
    print(f"freq tiles {n_tiles} starts={starts} width={nc}; time windows {len(t_starts)} "
          f"starts={t_starts} height={th} ({'crop' if th == sz else 'resize'}); "
          f"{n_cross} baselines -> {cap} units; "
          f"clean target: {'yes' if clean_target else 'no'}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fracs = []
    with h5py.File(out_path, 'w') as f:
        def mk(name, dt, last=None):
            sh = (cap,) if last is None else (cap, last, last)
            return f.create_dataset(name, shape=sh, dtype=dt,
                                    chunks=(1, last, last) if last else None)

        clean_ds = mk('clean', np.float32, sz)
        corr_ds  = mk('corrupted', np.float32, sz)
        mask_ds  = mk('mask', np.float32, sz)
        div_ds   = mk('dn_divisor', np.float32, sz)
        phase_ds = mk('phase', np.float32, sz)
        bl_ds    = mk('baseline_id', np.int32)
        a1_ds    = mk('ant1', np.int32)
        a2_ds    = mk('ant2', np.int32)
        tlo_ds   = mk('time_lo', np.int32)
        flo_ds   = mk('freq_lo', np.int32)
        ntd_ds   = mk('native_n_time', np.int32)
        ncd_ds   = mk('native_n_chan', np.int32)
        fmn_ds   = mk('freq_min_patch', np.float32)
        fmx_ds   = mk('freq_max_patch', np.float32)
        if clean_target:
            ampt_ds = mk('amp_target', np.float32, sz)
            pht_ds  = mk('phase_target', np.float32, sz)

        w = 0
        for u in range(n_cross):
            clean_n = fin['clean'][u]
            div_n   = fin['dn_divisor'][u]
            phase_n = fin['phase'][u]
            if clean_target:
                ampt_n = fin['amp_target'][u]
                pht_n  = fin['phase_target'][u]
            bl = int(fin['baseline_id'][u])
            a1 = int(fin['ant1'][u]); a2 = int(fin['ant2'][u])

            if controlled:
                corrupted_n, mask_n = controlled_inject(clean_n, spans)
            else:
                corrupted_n, mask_n = inject(clean_n, gen, synth_cfg, bands, args.target_frac,
                                             args.scale_min, args.scale_max, args.persist_frac)
            fracs.append(float(mask_n.mean()))

            for tlo in t_starts:
              for f0 in starts:
                f1 = f0 + nc
                t1 = tlo + th
                cl = resize_hw(clean_n[tlo:t1, f0:f1], sz, sz)
                co = resize_hw(corrupted_n[tlo:t1, f0:f1], sz, sz)
                dv = resize_hw(div_n[tlo:t1, f0:f1], sz, sz)
                ph = resize_phase_hw(phase_n[tlo:t1, f0:f1], sz, sz)
                mk_ = (resize_hw(mask_n[tlo:t1, f0:f1], sz, sz) > 0.5).astype(np.float32)

                clean_ds[w] = cl; corr_ds[w] = co; div_ds[w] = dv; phase_ds[w] = ph
                mask_ds[w] = mk_
                if clean_target:
                    ampt_ds[w] = resize_hw(ampt_n[tlo:t1, f0:f1], sz, sz)
                    pht_ds[w]  = resize_phase_hw(pht_n[tlo:t1, f0:f1], sz, sz)
                bl_ds[w] = bl; a1_ds[w] = a1; a2_ds[w] = a2
                tlo_ds[w] = tlo; flo_ds[w] = f0
                ntd_ds[w] = th; ncd_ds[w] = nc
                fmn_ds[w] = float(freqs[f0]); fmx_ds[w] = float(freqs[min(f1 - 1, full_n_chan - 1)])
                w += 1

            if (u + 1) % 200 == 0 or u == 0:
                rate = (u + 1) / max(time.time() - t0, 1e-6)
                print(f"  baseline {u + 1}/{n_cross}  units {w}/{cap}  "
                      f"mean flag frac {np.mean(fracs):.3f}  ({rate:.1f} bl/s)", flush=True)

        f.attrs['freq_min_mhz'] = freq_min
        f.attrs['freq_max_mhz'] = freq_max
        f.attrs['n_time'] = sz; f.attrs['n_freq'] = sz; f.attrs['img_size'] = sz
        f.attrs['n_baselines'] = cap
        f.attrs['full_n_time'] = full_n_time; f.attrs['full_n_chan'] = full_n_chan
        f.attrs['chan_lo'] = chan_lo
        f.attrs['seed'] = args.seed
        f.attrs['clean_target'] = clean_target

    fin.close()
    print(f"mean flag frac : {np.mean(fracs):.3f}  (target {args.target_frac})", flush=True)
    print(f"saved {cap} tiled units ({sz}x{sz}) -> {out_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--img-size',     type=int,   default=512)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--target-frac',  type=float, default=0.37)
    parser.add_argument('--band-width',   type=int,   default=0,
                        help='>0 = controlled test mode: deterministic full-time freq stripes of this '
                             'native-channel width (fraction ~ target-frac); 0 = original stochastic RFI')
    parser.add_argument('--persist-frac', type=float, default=0.6)
    parser.add_argument('--scale-min',    type=float, default=RFI_SCALE_MIN)
    parser.add_argument('--scale-max',    type=float, default=RFI_SCALE_MAX)
    main(parser.parse_args())
