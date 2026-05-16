import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


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

        n_plot = min(n_plot, n_samples)
        clean = f["clean"][:n_plot]
        corrupted = f["corrupted"][:n_plot]
        mask = f["mask"][:n_plot]

        all_clean = f["clean"][:500]
        all_mask = f["mask"][:500]

    flag_fractions = all_mask.mean(axis=(1, 2))
    print(f"\nflag fraction  mean={flag_fractions.mean():.3f}  "
          f"min={flag_fractions.min():.3f}  max={flag_fractions.max():.3f}")

    if flag_fractions.mean() < 0.05 or flag_fractions.mean() > 0.70:
        print("WARNING: flag fraction outside expected range 0.05–0.70")

    clean_mean = all_clean.mean()
    clean_std = all_clean.std()
    print(f"clean amp      mean={clean_mean:.4f}  std={clean_std:.4f}")

    if np.isnan(all_clean).any():
        print("WARNING: NaNs found in clean data")
    if np.isnan(f["corrupted"] if False else corrupted).any():
        print("WARNING: NaNs found in corrupted data")

    freqs = np.linspace(freq_min, freq_max, n_freq)
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
    mean_mask_spectrum = all_mask.mean(axis=(0, 1))

    axes[0].plot(freqs, mean_clean_spectrum)
    axes[0].set_xlabel("Freq (MHz)")
    axes[0].set_ylabel("Mean amplitude")
    axes[0].set_title("Mean clean spectrum")

    axes[1].plot(freqs, mean_mask_spectrum)
    axes[1].set_xlabel("Freq (MHz)")
    axes[1].set_ylabel("Flag fraction")
    axes[1].set_title("Mean flag fraction per channel")
    axes[1].axhline(0.2, color="r", linestyle="--", linewidth=0.8, label="20%")
    axes[1].axhline(0.3, color="orange", linestyle="--", linewidth=0.8, label="30%")
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
