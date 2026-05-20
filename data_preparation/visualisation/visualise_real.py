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


def extract_waterfall(ms_path, max_time, chan_avg=1):
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

    if chan_avg > 1:
        n_chan = waterfall.shape[1]
        n_trim = (n_chan // chan_avg) * chan_avg
        waterfall = waterfall[:, :n_trim].reshape(n_time, -1, chan_avg).mean(axis=2)
        all_flagged = all_flagged[:, :n_trim].reshape(n_time, -1, chan_avg).any(axis=2).astype(np.float32)
        freqs = freqs[:n_trim].reshape(-1, chan_avg).mean(axis=1)

    print(f"column: {col}")
    print(f"shape: {waterfall.shape}  freq: {freqs[0]:.1f}-{freqs[-1]:.1f} MHz")
    print(f"channel averaging: {chan_avg}x  ({freqs[1]-freqs[0]:.2f} MHz/bin)")
    print(f"flagged cells: {all_flagged.mean()*100:.1f}%")

    return waterfall, all_flagged.astype(np.float32), freqs


def in_rfi_band(freqs):
    mask = np.zeros(len(freqs), dtype=bool)
    for lo, hi in RFI_BANDS:
        mask |= (freqs >= lo) & (freqs <= hi)
    return mask


def clean_percentile(waterfall, flag_mask, freqs, lo=5, hi=99):
    rfi_chans = in_rfi_band(freqs)
    clean = waterfall[:, ~rfi_chans]
    clean_flags = flag_mask[:, ~rfi_chans]
    vals = clean[clean_flags == 0]
    return np.percentile(vals, lo), np.percentile(vals, hi)


def green_overlay(waterfall_2d, flag_mask_2d, threshold=1.0):
    combined = (flag_mask_2d > 0) | (waterfall_2d >= threshold)
    rgba = np.zeros((*combined.T.shape, 4), dtype=np.float32)
    rgba[combined.T] = [0.0, 1.0, 0.2, 1.0]
    return rgba


def add_rfi_bands_y(ax, freqs):
    for lo, hi in RFI_BANDS:
        if lo < freqs[-1] and hi > freqs[0]:
            ax.axhspan(max(lo, freqs[0]), min(hi, freqs[-1]),
                       color='cyan', alpha=0.15, linewidth=0)


def extract_patches(waterfall, flag_mask, freqs, n_patches=16):
    n_time, n_chan = waterfall.shape
    if n_time < PATCH_SIZE or n_chan < PATCH_SIZE:
        raise ValueError(f"waterfall {waterfall.shape} too small for {PATCH_SIZE}x{PATCH_SIZE} patches")

    # tile across frequency — 4096 channels gives 16 non-overlapping freq slices
    # fix time window to the first PATCH_SIZE bins
    t0 = 0
    patches, patch_flags, patch_freqs, patch_times = [], [], [], []
    freq_step = max(PATCH_SIZE, (n_chan - PATCH_SIZE) // max(1, n_patches - 1))
    for i in range(n_patches):
        f0 = min(i * freq_step, n_chan - PATCH_SIZE)
        patches.append(waterfall[t0:t0 + PATCH_SIZE, f0:f0 + PATCH_SIZE])
        patch_flags.append(flag_mask[t0:t0 + PATCH_SIZE, f0:f0 + PATCH_SIZE])
        patch_freqs.append(freqs[f0:f0 + PATCH_SIZE])
        patch_times.append((t0, t0 + PATCH_SIZE))

    return patches, patch_flags, patch_freqs, patch_times


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading MS...")
    waterfall, flag_mask, freqs = extract_waterfall(args.ms, args.max_time, args.chan_avg)

    vmin, _ = clean_percentile(waterfall, flag_mask, freqs, 1, 99)
    vmax = 1.0

    rfi_chans = in_rfi_band(freqs)
    unflagged_clean = waterfall[:, ~rfi_chans][flag_mask[:, ~rfi_chans] == 0]

    # --- 256x256 patches with flag overlay ---
    patches, patch_flags, patch_freqs, patch_times = extract_patches(waterfall, flag_mask, freqs)
    n = len(patches)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, (patch, pflags, pf, pt) in enumerate(zip(patches, patch_flags, patch_freqs, patch_times)):
        ax = axes[i]
        ax.imshow(patch.T, aspect='auto', origin='lower',
                  extent=[pt[0], pt[1], pf[0], pf[-1]],
                  vmin=vmin, vmax=vmax, cmap='plasma')
        ax.imshow(green_overlay(patch, pflags), aspect='auto', origin='lower',
                  extent=[pt[0], pt[1], pf[0], pf[-1]])
        add_rfi_bands_y(ax, pf)
        ax.set_title(f"t={pt[0]}–{pt[1]}", fontsize=8)
        ax.set_xlabel("Time bins", fontsize=7)
        ax.tick_params(labelsize=6)
        if i % ncols == 0:
            ax.set_ylabel("Freq (MHz)", fontsize=7)
        else:
            ax.set_yticklabels([])
    for ax in axes[n:]:
        ax.set_visible(False)
    plt.suptitle("Real MeerKAT — 256×256 patches (green = flagged)", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "patches.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved patches.png")

    # --- Amplitude distribution (non-RFI-band, unflagged only) ---
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

    # --- Full waterfall overview with flag overlay ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(waterfall.T, aspect='auto', origin='lower',
              extent=[0, waterfall.shape[0], freqs[0], freqs[-1]],
              vmin=vmin, vmax=vmax, cmap='plasma')
    ax.imshow(green_overlay(waterfall, flag_mask), aspect='auto', origin='lower',
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
    parser.add_argument('--chan-avg', type=int, default=16)
    main(parser.parse_args())
