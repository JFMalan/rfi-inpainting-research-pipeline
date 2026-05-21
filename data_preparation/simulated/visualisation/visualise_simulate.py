import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def green_overlay(mask_2d):
    rgba = np.zeros((*mask_2d.T.shape, 4), dtype=np.float32)
    rgba[mask_2d.T > 0] = [0.0, 1.0, 0.2, 0.85]
    return rgba


def run_checks(path, n_check=500):
    with h5py.File(path, "r") as f:
        n_samples = f["clean"].shape[0]
        n_time    = int(f.attrs["n_time"])
        n_freq    = int(f.attrs["n_freq"])
        freq_min  = float(f.attrs["freq_min_mhz"])
        freq_max  = float(f.attrs["freq_max_mhz"])
        n_check   = min(n_check, n_samples)
        all_clean     = f["clean"][:n_check]
        all_corrupted = f["corrupted"][:n_check]
        all_mask      = f["mask"][:n_check]

    freqs = np.linspace(freq_min, freq_max, n_freq)
    ok = True

    print(f"samples : {n_samples}")
    print(f"shape   : {n_time} time x {n_freq} freq")
    print(f"freq    : {freq_min:.1f}–{freq_max:.1f} MHz")

    flag_fractions = all_mask.mean(axis=(1, 2))
    print(f"\nflag fraction  mean={flag_fractions.mean():.3f}  "
          f"min={flag_fractions.min():.3f}  max={flag_fractions.max():.3f}")
    if not (0.05 <= flag_fractions.mean() <= 0.70):
        print("WARNING: mean flag fraction outside expected range 0.05–0.70")
        ok = False

    if np.isnan(all_clean).any():
        print("WARNING: NaNs in clean data")
        ok = False
    if np.isnan(all_corrupted).any():
        print("WARNING: NaNs in corrupted data")
        ok = False

    rfi_only = all_corrupted - all_clean
    mean_rfi_at_mask = rfi_only[all_mask > 0].mean() if (all_mask > 0).any() else 0
    print(f"\nRFI amplitude check:")
    print(f"  mean clean amp       : {all_clean.mean():.4f}")
    print(f"  clean amp std        : {all_clean.std():.4f}")
    print(f"  mean RFI at mask     : {mean_rfi_at_mask:.4f}")
    if mean_rfi_at_mask <= 0:
        print("WARNING: RFI amplitude at masked pixels is not positive")
        ok = False

    mean_mask_spectrum = all_mask.mean(axis=(0, 1))
    mean_mask_time     = all_mask.mean(axis=(0, 2))
    print(f"\nmorphology check:")
    print(f"  max per-channel flag fraction : {mean_mask_spectrum.max():.3f}")
    print(f"  max per-time flag fraction    : {mean_mask_time.max():.3f}")
    if mean_mask_spectrum.max() < 0.15:
        print("WARNING: no strongly flagged channels — narrowband RFI may be missing")
        ok = False
    if mean_mask_time.max() < 0.10:
        print("WARNING: no strongly flagged time steps — broadband/bursty RFI may be missing")
        ok = False

    n_run_check = min(50, n_check)
    mean_run_lengths = []
    for i in range(n_run_check):
        col = all_mask[i, :, all_mask[i].mean(axis=0).argmax()]
        if col.sum() == 0:
            continue
        runs, run_len = [], 0
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

    return all_clean, all_corrupted, all_mask, freqs, freq_min, freq_max, n_time, n_freq, mean_mask_spectrum, ok


def plot_sample_pairs(clean, corrupted, mask, freqs, freq_min, freq_max, n_time, n_plot, out_dir):
    n_plot = min(n_plot, len(clean))
    ncols = 3
    fig, axes = plt.subplots(n_plot, ncols, figsize=(12, 3.5 * n_plot))
    if n_plot == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_plot):
        vmin = np.percentile(clean[i], 1)
        vmax = np.percentile(clean[i], 99)

        axes[i, 0].imshow(clean[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=vmin, vmax=vmax, cmap="plasma")
        axes[i, 0].set_ylabel("Freq (MHz)", fontsize=8)
        if i == 0:
            axes[i, 0].set_title("clean")

        axes[i, 1].imshow(corrupted[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=vmin, vmax=vmax, cmap="plasma")
        axes[i, 1].imshow(green_overlay(mask[i]), aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max])
        if i == 0:
            axes[i, 1].set_title("corrupted (green = RFI mask)")

        axes[i, 2].imshow(mask[i].T, aspect="auto", origin="lower",
                          extent=[0, n_time, freq_min, freq_max],
                          vmin=0, vmax=1, cmap="binary_r")
        if i == 0:
            axes[i, 2].set_title("mask")

    for ax in axes[-1]:
        ax.set_xlabel("Time bins", fontsize=8)
    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = out_dir / "validate_samples.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"samples plot    -> {out}")


def plot_amplitude_dist(clean, out_dir):
    vals = clean.ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.percentile(vals, 0.5), np.percentile(vals, 99.5), 120)
    ax.hist(vals, bins=bins, density=True, color='steelblue', alpha=0.8)
    ax.axvline(vals.mean(), color='red', linewidth=1, label=f"mean={vals.mean():.4f}")
    ax.axvline(np.median(vals), color='orange', linewidth=1, linestyle='--',
               label=f"median={np.median(vals):.4f}")
    ax.set_xlabel("Amplitude (div-norm units)")
    ax.set_ylabel("Density")
    ax.set_title("Simulated — clean patch amplitude distribution")
    ax.legend()
    plt.tight_layout()
    out = out_dir / "amplitude_dist.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"amplitude dist  -> {out}")


def plot_spectra(clean, mean_mask_spectrum, freqs, out_dir):
    mean_clean_spectrum = clean.mean(axis=(0, 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(freqs, mean_clean_spectrum, linewidth=0.6, color='steelblue')
    axes[0].set_yscale('log')
    axes[0].set_xlabel("Freq (MHz)")
    axes[0].set_ylabel("Mean amplitude (log scale)")
    axes[0].set_title("Simulated — mean clean spectrum")

    axes[1].plot(freqs, mean_mask_spectrum, linewidth=0.8, color='steelblue')
    axes[1].axhline(0.10, color='red', linestyle='--', linewidth=0.8, label='10%')
    axes[1].set_xlabel("Freq (MHz)")
    axes[1].set_ylabel("Flag fraction")
    axes[1].set_title("Mean RFI mask fraction per channel")
    axes[1].legend()

    plt.tight_layout()
    out = out_dir / "validate_spectra.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"spectra plot    -> {out}")


def plot_waterfall(waterfall_path, out_dir):
    waterfall = np.load(waterfall_path)
    meta      = np.load(waterfall_path.replace('.npy', '.meta.npy'))
    flags_path = waterfall_path.replace('.npy', '.flags.npy')
    flag_map  = np.load(flags_path) if Path(flags_path).exists() else np.zeros(waterfall.shape, dtype=bool)

    freq_min, freq_max = float(meta[0]), float(meta[1])
    freqs = np.linspace(freq_min, freq_max, waterfall.shape[1])

    unflagged = waterfall[flag_map == 0]
    vmin = np.percentile(unflagged, 1)
    vmax = np.percentile(unflagged, 99)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(waterfall.T, aspect='auto', origin='lower',
                   extent=[0, waterfall.shape[0], freq_min, freq_max],
                   vmin=vmin, vmax=vmax, cmap='plasma')
    ax.imshow(green_overlay(flag_map), aspect='auto', origin='lower',
              extent=[0, waterfall.shape[0], freq_min, freq_max])
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Freq (MHz)")
    ax.set_title("Simulated — baseline-averaged waterfall (green = flagged)")
    plt.colorbar(im, ax=ax, label="Amplitude (div-norm units)", pad=0.01)
    plt.tight_layout()
    out = out_dir / "waterfall_full.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"waterfall       -> {out}")


def plot_patch_pages(path, out_dir, n_show, per_page=16):
    with h5py.File(path, "r") as f:
        n_total = f["corrupted"].shape[0]
        n_time  = int(f.attrs["n_time"])
        n_freq  = int(f.attrs["n_freq"])
        freq_min = float(f.attrs["freq_min_mhz"])
        freq_max = float(f.attrs["freq_max_mhz"])
        indices   = np.linspace(0, n_total - 1, min(n_show, n_total), dtype=int)
        corrupted = f["corrupted"][indices]
        masks     = f["mask"][indices]
        if "freq_min_patch" in f:
            patch_fmin = f["freq_min_patch"][:][indices]
            patch_fmax = f["freq_max_patch"][:][indices]
        else:
            patch_fmin = np.full(len(indices), freq_min)
            patch_fmax = np.full(len(indices), freq_max)

    patch_dir = out_dir / "patches_hdf5"
    patch_dir.mkdir(exist_ok=True)

    ncols  = 4
    pages  = (len(indices) + per_page - 1) // per_page

    for page in range(pages):
        sl = slice(page * per_page, (page + 1) * per_page)
        pg_corrupted = corrupted[sl]
        pg_masks     = masks[sl]
        pg_indices   = indices[sl]
        pg_fmin      = patch_fmin[sl]
        pg_fmax      = patch_fmax[sl]
        n = len(pg_corrupted)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
        axes = np.array(axes).flatten()

        for i in range(n):
            patch = pg_corrupted[i]
            fm    = pg_masks[i]
            fmin  = pg_fmin[i]
            fmax  = pg_fmax[i]
            clean_pixels = patch[fm == 0]
            vmin = np.percentile(clean_pixels, 2) if len(clean_pixels) > 10 else patch.min()
            vmax = np.percentile(clean_pixels, 98) if len(clean_pixels) > 10 else patch.max()
            ax = axes[i]
            ax.imshow(patch.T, aspect='auto', origin='lower',
                      extent=[0, n_time, fmin, fmax],
                      vmin=vmin, vmax=vmax, cmap='plasma')
            ax.imshow(green_overlay(fm), aspect='auto', origin='lower',
                      extent=[0, n_time, fmin, fmax])
            ax.set_title(f"patch {pg_indices[i]}  rfi={fm.mean():.2f}", fontsize=8)
            ax.tick_params(labelsize=6)
            if i % ncols == 0:
                ax.set_ylabel("Freq (MHz)", fontsize=7)
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel("Time bins", fontsize=7)

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.suptitle(
            f"Simulated — corrupted patches (green = RFI mask)  "
            f"[page {page + 1}/{pages}, {n_total} total]",
            y=1.01)
        plt.tight_layout()
        out_path = patch_dir / f"page_{page + 1:03d}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()

    print(f"patch pages     -> {patch_dir}/  ({pages} pages, {len(indices)} of {n_total} shown)")


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean, corrupted, mask, freqs, freq_min, freq_max, n_time, n_freq, mean_mask_spectrum, _ = \
        run_checks(args.input)

    plot_sample_pairs(clean, corrupted, mask, freqs, freq_min, freq_max, n_time, args.n_plot, out_dir)
    plot_amplitude_dist(clean, out_dir)
    plot_spectra(clean, mean_mask_spectrum, freqs, out_dir)
    if args.waterfall:
        plot_waterfall(args.waterfall, out_dir)
    plot_patch_pages(args.input, out_dir, args.n_patches_show)

    print(f"\nall plots -> {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",           required=True)
    parser.add_argument("--output",          required=True)
    parser.add_argument("--n-plot",          type=int,   default=12,   dest="n_plot")
    parser.add_argument("--n-patches-show",  type=int,   default=200,  dest="n_patches_show")
    parser.add_argument("--waterfall",       default=None)
    main(parser.parse_args())
