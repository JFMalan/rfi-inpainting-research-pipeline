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
    if not (0.05 <= flag_fractions.mean() <= 0.70):
        print("WARNING: mean flag fraction outside expected range 0.05–0.70")
        ok = False

    # --- RFI present check: corrupted > clean where masked ---
    mean_clean = all_clean[all_mask > 0].mean() if (all_mask > 0).any() else 0
    mean_corrupted_at_mask = (all_clean + (all_mask > 0).astype(np.float32)).mean()
    rfi_excess = (all_clean * (all_mask == 0)).mean()
    print(f"\nRFI amplitude check:")
    print(f"  mean clean amp          : {all_clean.mean():.4f}")
    print(f"  clean amp std           : {all_clean.std():.4f}")
    if np.isnan(all_clean).any():
        print("WARNING: NaNs in clean data")
        ok = False

    corrupted_all = None
    with h5py.File(path, "r") as f:
        corrupted_all = f["corrupted"][:n_check]
    rfi_only = corrupted_all - all_clean
    mean_rfi_at_mask = rfi_only[all_mask > 0].mean() if (all_mask > 0).any() else 0
    print(f"  mean RFI amp at mask    : {mean_rfi_at_mask:.4f}")
    if mean_rfi_at_mask <= 0:
        print("WARNING: RFI amplitude at masked pixels is not positive")
        ok = False
    if np.isnan(corrupted_all).any():
        print("WARNING: NaNs in corrupted data")
        ok = False

    # --- Morphology check: at least some samples have each axis type ---
    # Check for narrowband (column-dominant) and broadband (row-dominant) RFI
    mean_mask_spectrum = all_mask.mean(axis=(0, 1))   # (n_freq,)
    mean_mask_time = all_mask.mean(axis=(0, 2))        # (n_time,)
    print(f"\nmorphology check:")
    print(f"  max per-channel flag fraction : {mean_mask_spectrum.max():.3f}")
    print(f"  max per-time flag fraction    : {mean_mask_time.max():.3f}")
    if mean_mask_spectrum.max() < 0.15:
        print("WARNING: no strongly flagged channels — narrowband RFI may be missing")
        ok = False
    if mean_mask_time.max() < 0.10:
        print("WARNING: no strongly flagged time steps — broadband/bursty RFI may be missing")
        ok = False

    # --- Temporal contiguity: mask should have runs > 1 bin ---
    n_run_check = min(50, n_check)
    mean_run_lengths = []
    for i in range(n_run_check):
        col = all_mask[i, :, all_mask[i].mean(axis=0).argmax()]
        if col.sum() == 0:
            continue
        runs = []
        run_len = 0
        for v in col:
            if v > 0:
                run_len += 1
            elif run_len > 0:
                runs.append(run_len)
                run_len = 0
        if run_len > 0:
            runs.append(run_len)
        if runs:
            mean_run_lengths.append(np.mean(runs))
    if mean_run_lengths:
        avg_run = np.mean(mean_run_lengths)
        print(f"\ntemporal run length (most-flagged channel): mean={avg_run:.1f} bins")
        if avg_run < 2.0:
            print("WARNING: mean run length < 2 bins — RFI may be too sparse")
            ok = False

    print(f"\nvalidation {'PASSED' if ok else 'FAILED (see warnings above)'}")

    # --- Plots ---
    out_dir = Path(path).parent

    fig, axes = plt.subplots(n_plot, 3, figsize=(12, 3 * n_plot))
    for i in range(n_plot):
        vmin_c = np.percentile(clean[i], 1)
        vmax_c = np.percentile(clean[i], 99)
        vmax_cor = np.percentile(corrupted[i], 99)
        axes[i, 0].imshow(clean[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=vmin_c, vmax=vmax_c, cmap="plasma")
        axes[i, 0].set_ylabel("Freq (MHz)")
        if i == 0:
            axes[i, 0].set_title("clean")
        axes[i, 1].imshow(corrupted[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=vmin_c, vmax=vmax_c, cmap="plasma")
        rfi_rgba = np.zeros((*mask[i].T.shape, 4), dtype=np.float32)
        rfi_rgba[mask[i].T > 0] = [0.0, 1.0, 0.2]
        axes[i, 1].imshow(rfi_rgba, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max])
        if i == 0:
            axes[i, 1].set_title("corrupted")
        axes[i, 2].imshow(mask[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=0, vmax=1, cmap="binary_r")
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
    axes[1].set_xlabel("Freq (MHz)")
    axes[1].set_ylabel("Flag fraction")
    axes[1].set_title("Mean flag fraction per channel")
    axes[1].axhline(0.10, color="r", linestyle="--", linewidth=0.8, label="10%")
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
