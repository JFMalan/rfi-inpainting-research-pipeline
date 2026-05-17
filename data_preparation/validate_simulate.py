import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


GNSS_LINES_MHZ = [1176.45, 1207.14, 1227.60, 1246.0, 1575.42, 1602.0]

PERSISTENT_BANDS = [(930.0, 960.0), (1163.0, 1299.0), (1530.0, 1630.0)]

CLEAN_BANDS = [(880.0, 933.0), (960.0, 1163.0), (1299.0, 1524.0), (1630.0, 1680.0)]


def check_dataset(path, n_plot=4):
    with h5py.File(path, "r") as f:
        n_samples = f["clean"].shape[0]
        n_time = f.attrs["n_time"]
        n_freq = f.attrs["n_freq"]
        freq_min = f.attrs["freq_min_mhz"]
        freq_max = f.attrs["freq_max_mhz"]

        print(f"samples : {n_samples}")
        print(f"shape   : {n_time} time x {n_freq} freq")
        print(f"freq    : {freq_min}–{freq_max} MHz")

        n_check = min(500, n_samples)
        n_plot = min(n_plot, n_samples)
        clean = f["clean"][:n_plot]
        corrupted = f["corrupted"][:n_plot]
        mask = f["mask"][:n_plot]
        all_clean = f["clean"][:n_check]
        all_mask = f["mask"][:n_check]

    freqs = np.linspace(freq_min, freq_max, n_freq)
    ok = True

    # --- Overall flag fraction ---
    flag_fractions = all_mask.mean(axis=(1, 2))
    print(f"\nflag fraction  mean={flag_fractions.mean():.3f}  "
          f"min={flag_fractions.min():.3f}  max={flag_fractions.max():.3f}")
    # MeerKLASS DR1 (Cunnington et al. 2025): 40-50% post-tricolour L-band flag fraction
    if not (0.20 <= flag_fractions.mean() <= 0.70):
        print("WARNING: mean flag fraction outside expected range 0.20–0.70")
        ok = False

    # --- Per-band flag fractions ---
    print("\nper-band flag fractions:")
    mean_mask_spectrum = all_mask.mean(axis=(0, 1))

    for f_lo, f_hi in PERSISTENT_BANDS:
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if band.sum() == 0:
            continue
        frac = mean_mask_spectrum[band].mean()
        print(f"  {f_lo:.0f}–{f_hi:.0f} MHz (persistent RFI): {frac:.3f}")
        if frac < 0.40:
            print(f"  WARNING: persistent RFI band {f_lo:.0f}–{f_hi:.0f} MHz flag fraction {frac:.3f} < 0.40")
            ok = False

    for f_lo, f_hi in CLEAN_BANDS:
        band = (freqs >= f_lo) & (freqs <= f_hi)
        if band.sum() == 0:
            continue
        frac = mean_mask_spectrum[band].mean()
        print(f"  {f_lo:.0f}–{f_hi:.0f} MHz (clean band):     {frac:.3f}")
        # Sihlangu et al. (2022): <2% occupancy in clean bands from persistent sources
        # We allow up to 10% for the narrowband emitters in our model
        if frac > 0.10:
            print(f"  WARNING: clean band {f_lo:.0f}–{f_hi:.0f} MHz flag fraction {frac:.3f} > 0.10")
            ok = False

    # --- GNSS spectral line peaks ---
    print("\nGNSS line peak check:")
    for line_mhz in GNSS_LINES_MHZ:
        window = (freqs >= line_mhz - 15) & (freqs <= line_mhz + 15)
        if window.sum() == 0:
            continue
        peak = mean_mask_spectrum[window].max()
        print(f"  {line_mhz:.2f} MHz: peak flag fraction in ±15 MHz window = {peak:.3f}")
        if peak < 0.30:
            print(f"  WARNING: no clear RFI peak near {line_mhz:.2f} MHz (peak={peak:.3f})")
            ok = False

    # --- Amplitude statistics ---
    clean_mean = all_clean.mean()
    clean_std = all_clean.std()
    print(f"\nclean amp      mean={clean_mean:.4f}  std={clean_std:.4f}")

    if np.isnan(all_clean).any():
        print("WARNING: NaNs found in clean data")
        ok = False
    if np.isnan(corrupted).any():
        print("WARNING: NaNs found in corrupted data")
        ok = False

    # --- Temporal structure check: mask should have contiguous blocks ---
    # Sample a few patches and check that mask runs in time > 1 bin exist
    n_run_check = min(50, n_check)
    mean_run_lengths = []
    for i in range(n_run_check):
        col = all_mask[i, :, all_mask[i].mean(axis=0).argmax()]
        if col.sum() == 0:
            continue
        runs = []
        in_run = False
        run_len = 0
        for v in col:
            if v > 0:
                run_len += 1
                in_run = True
            elif in_run:
                runs.append(run_len)
                run_len = 0
                in_run = False
        if in_run:
            runs.append(run_len)
        if runs:
            mean_run_lengths.append(np.mean(runs))
    if mean_run_lengths:
        avg_run = np.mean(mean_run_lengths)
        print(f"\ntemporal run length (most-flagged channel): mean={avg_run:.1f} bins")
        if avg_run < 3.0:
            print("WARNING: mean run length < 3 bins — RFI may be too bursty (expected contiguous blocks)")
            ok = False

    print(f"\nvalidation {'PASSED' if ok else 'FAILED (see warnings above)'}")

    # --- Plots ---
    out_dir = Path(path).parent

    fig, axes = plt.subplots(n_plot, 3, figsize=(12, 3 * n_plot))
    for i in range(n_plot):
        vmax_clean = np.percentile(clean[i], 99)
        vmax_corr = np.percentile(corrupted[i], 99)
        axes[i, 0].imshow(clean[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=0, vmax=vmax_clean, cmap="viridis")
        axes[i, 0].set_ylabel("Freq (MHz)")
        if i == 0:
            axes[i, 0].set_title("clean")
        axes[i, 1].imshow(corrupted[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=0, vmax=vmax_corr, cmap="viridis")
        if i == 0:
            axes[i, 1].set_title("corrupted")
        axes[i, 2].imshow(mask[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=0, vmax=1, cmap="Reds")
        if i == 0:
            axes[i, 2].set_title("mask")
    for ax in axes[-1]:
        ax.set_xlabel("Time bins")
    plt.tight_layout()
    plot_path = out_dir / "validate_samples.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"\nsamples plot -> {plot_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mean_clean_spectrum = all_clean.mean(axis=(0, 1))
    axes[0].plot(freqs, mean_clean_spectrum)
    axes[0].set_xlabel("Freq (MHz)")
    axes[0].set_ylabel("Mean amplitude")
    axes[0].set_title("Mean clean spectrum")

    axes[1].plot(freqs, mean_mask_spectrum, linewidth=0.8)
    for line_mhz in GNSS_LINES_MHZ:
        if freq_min <= line_mhz <= freq_max:
            axes[1].axvline(line_mhz, color="green", linewidth=0.6, alpha=0.7)
    axes[1].set_xlabel("Freq (MHz)")
    axes[1].set_ylabel("Flag fraction")
    axes[1].set_title("Mean flag fraction per channel  (green = GNSS carriers)")
    axes[1].axhline(0.40, color="r", linestyle="--", linewidth=0.8, label="40%")
    axes[1].legend()
    plt.tight_layout()
    spec_path = out_dir / "validate_spectra.png"
    plt.savefig(spec_path, dpi=120)
    plt.close()
    print(f"spectra plot  -> {spec_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--n_plot", type=int, default=12)
    args = parser.parse_args()
    check_dataset(args.input, args.n_plot)
