import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from casacore.tables import table

RFI_BANDS = [
    (930, 960),
    (1170, 1300),
    (1525, 1630),
]

PATCH_SIZE = 256


def extract_waterfall(ms_path, max_time):
    ms = table(ms_path, readonly=True)
    cols = ms.colnames()
    col = 'CORRECTED_DATA' if 'CORRECTED_DATA' in cols else 'DATA'
    try:
        ms.getcell(col, 0)
    except Exception:
        col = 'DATA'

    data = ms.getcol(col)
    flags = ms.getcol('FLAG')
    times = ms.getcol('TIME')
    ms.close()

    freqs_table = table(ms_path + '/SPECTRAL_WINDOW')
    freqs = freqs_table.getcol('CHAN_FREQ')[0] / 1e6
    freqs_table.close()

    amp = np.abs(data).mean(axis=2).astype(np.float32)
    flagged = flags.any(axis=2)

    unique_times = np.unique(times)
    n_baseline = amp.shape[0] // len(unique_times)
    n_time = min(len(unique_times), max_time)

    amp = amp[:n_time * n_baseline]
    flagged = flagged[:n_time * n_baseline]

    amp_3d = np.ma.array(
        amp.reshape(n_time, n_baseline, amp.shape[1]),
        mask=flagged.reshape(n_time, n_baseline, amp.shape[1])
    )
    waterfall = amp_3d.mean(axis=1).filled(0.0).astype(np.float32)
    all_flagged = flagged.reshape(n_time, n_baseline, amp.shape[1]).all(axis=1)

    print(f"column: {col}")
    print(f"shape: {waterfall.shape}  freq: {freqs[0]:.1f}-{freqs[-1]:.1f} MHz")
    print(f"flagged cells: {all_flagged.mean()*100:.1f}%")

    return waterfall, all_flagged.astype(np.float32), freqs


def in_rfi_band(freqs):
    mask = np.zeros(len(freqs), dtype=bool)
    for lo, hi in RFI_BANDS:
        mask |= (freqs >= lo) & (freqs <= hi)
    return mask


def green_overlay(flag_mask_2d):
    rgba = np.zeros((*flag_mask_2d.T.shape, 4), dtype=np.float32)
    rgba[flag_mask_2d.T > 0] = [0.0, 1.0, 0.2, 1.0]
    return rgba


def add_rfi_bands_y(ax, freqs):
    for lo, hi in RFI_BANDS:
        if lo < freqs[-1] and hi > freqs[0]:
            ax.axhspan(max(lo, freqs[0]), min(hi, freqs[-1]),
                       color='cyan', alpha=0.15, linewidth=0)


def extract_patches(waterfall, flag_mask, n_patches=16):
    n_time, n_chan = waterfall.shape
    if n_time < PATCH_SIZE:
        raise ValueError(f"not enough time bins ({n_time}) — increase --max-time")

    # only produce as many non-overlapping patches as the data supports
    max_non_overlapping = n_time // PATCH_SIZE
    n_patches = min(n_patches, max_non_overlapping)

    patches, patch_flags, patch_times = [], [], []
    for i in range(n_patches):
        t0 = i * PATCH_SIZE
        patches.append(waterfall[t0:t0 + PATCH_SIZE, :])
        patch_flags.append(flag_mask[t0:t0 + PATCH_SIZE, :])
        patch_times.append((t0, t0 + PATCH_SIZE))

    return patches, patch_flags, patch_times


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading MS...")
    waterfall, flag_mask, freqs = extract_waterfall(args.ms, args.max_time)

    rfi_chans = in_rfi_band(freqs)
    unflagged_clean = waterfall[:, ~rfi_chans][flag_mask[:, ~rfi_chans] == 0]
    global_vmin = np.percentile(unflagged_clean, 5)
    global_vmax = np.percentile(unflagged_clean, 95)

    # --- 16 patches tiled across frequency, per-patch colour scaling ---
    patches, patch_flags, patch_times = extract_patches(waterfall, flag_mask)
    n = len(patches)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 8 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, (patch, pflags, pt) in enumerate(zip(patches, patch_flags, patch_times)):
        ax = axes[i]
        im = ax.imshow(patch.T, aspect='auto', origin='lower',
                       extent=[pt[0], pt[1], freqs[0], freqs[-1]],
                       vmin=global_vmin, vmax=global_vmax, cmap='plasma')
        ax.imshow(green_overlay(pflags), aspect='auto', origin='lower',
                  extent=[pt[0], pt[1], freqs[0], freqs[-1]])
        add_rfi_bands_y(ax, freqs)
        ax.set_title(f"Real MeerKAT — t={pt[0]}–{pt[1]} (green = flagged)", fontsize=11)
        ax.set_xlabel("Time bins", fontsize=10)
        ax.set_ylabel("Freq (MHz)", fontsize=10)
        ax.tick_params(labelsize=9)
        plt.colorbar(im, ax=ax, label="Amplitude (Jy)", pad=0.01)
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "patches.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved patches.png")

    # --- Amplitude distribution (non-RFI, unflagged) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.percentile(unflagged_clean, 0.5),
                       np.percentile(unflagged_clean, 99.5), 120)
    ax.hist(unflagged_clean, bins=bins, density=True, color='steelblue', alpha=0.8)
    ax.axvline(unflagged_clean.mean(), color='red', linewidth=1,
               label=f"mean={unflagged_clean.mean():.4f}")
    ax.axvline(np.median(unflagged_clean), color='orange', linewidth=1, linestyle='--',
               label=f"median={np.median(unflagged_clean):.4f}")
    ax.set_xlabel("Amplitude (Jy)")
    ax.set_ylabel("Density")
    ax.set_title("Real MeerKAT — amplitude distribution (unflagged, non-RFI channels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_dist.png", dpi=120)
    plt.close()
    print("saved amplitude_dist.png")

    # --- Mean spectrum (log scale) ---
    mean_spec = np.where(flag_mask == 0, waterfall, np.nan)
    with np.errstate(all='ignore'):
        mean_spec = np.nanmean(mean_spec, axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freqs, mean_spec, linewidth=0.6, color='steelblue')
    ax.set_yscale('log')
    for lo, hi in RFI_BANDS:
        if lo < freqs[-1] and hi > freqs[0]:
            ax.axvspan(lo, hi, color='red', alpha=0.15, linewidth=0)
            mid = (max(lo, freqs[0]) + min(hi, freqs[-1])) / 2
            ax.text(mid, ax.get_ylim()[1], f"{lo}–{hi}", ha='center', va='bottom',
                    fontsize=7, color='red', rotation=90)
    ax.set_xlabel("Freq (MHz)")
    ax.set_ylabel("Mean amplitude (Jy, log scale)")
    ax.set_title("Real MeerKAT — mean spectrum (RFI bands shaded)")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_spectrum.png", dpi=120)
    plt.close()
    print("saved mean_spectrum.png")

    # --- Full waterfall ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(waterfall.T, aspect='auto', origin='lower',
              extent=[0, waterfall.shape[0], freqs[0], freqs[-1]],
              vmin=global_vmin, vmax=global_vmax, cmap='plasma')
    ax.imshow(green_overlay(flag_mask), aspect='auto', origin='lower',
              extent=[0, waterfall.shape[0], freqs[0], freqs[-1]])
    for lo, hi in RFI_BANDS:
        if lo < freqs[-1] and hi > freqs[0]:
            ax.axhspan(max(lo, freqs[0]), min(hi, freqs[-1]),
                       color='cyan', alpha=0.12, linewidth=0)
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Freq (MHz)")
    ax.set_title("Real MeerKAT — full waterfall (green = flagged)")
    plt.colorbar(ax.images[0], ax=ax, label="Amplitude (Jy)", pad=0.01)
    plt.tight_layout()
    plt.savefig(out_dir / "waterfall_full.png", dpi=100)
    plt.close()
    print("saved waterfall_full.png")

    print(f"\nstats (unflagged, non-RFI channels):")
    print(f"  mean={unflagged_clean.mean():.4f}  std={unflagged_clean.std():.4f}")
    print(f"  p5={np.percentile(unflagged_clean,5):.4f}  p95={np.percentile(unflagged_clean,95):.4f} Jy")
    print(f"\nall plots -> {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-time', type=int, default=512)
    main(parser.parse_args())
