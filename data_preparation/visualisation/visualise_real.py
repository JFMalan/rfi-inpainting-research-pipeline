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


def load_ms(ms_path, max_time, field=None):
    ms = table(ms_path, readonly=True)
    if field is not None:
        ms = ms.query(f"FIELD_ID == {field}")
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
    n_chan = amp.shape[1]
    n_time = min(len(unique_times), max_time)

    amp = amp[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)
    flagged = flagged[:n_time * n_baseline].reshape(n_time, n_baseline, n_chan)

    print(f"column: {col}  shape: ({n_time}, {n_baseline}, {n_chan})  "
          f"freq: {freqs[0]:.1f}-{freqs[-1]:.1f} MHz")

    return amp, flagged, freqs


def get_baseline_waterfall(amp, flagged, baseline_idx):
    w = amp[:, baseline_idx, :]
    f = flagged[:, baseline_idx, :]
    return w, f.astype(np.float32)


def get_avg_waterfall(amp, flagged):
    wf = np.ma.array(amp, mask=flagged).mean(axis=1).filled(0.0).astype(np.float32)
    fm = flagged.all(axis=1).astype(np.float32)
    return wf, fm


def green_overlay(flag_mask_2d):
    rgba = np.zeros((*flag_mask_2d.T.shape, 4), dtype=np.float32)
    rgba[flag_mask_2d.T > 0] = [0.0, 1.0, 0.2, 1]
    return rgba


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading MS...")
    amp, flagged, freqs = load_ms(args.ms, args.max_time, field=args.field)

    chan_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    amp = amp[:, :, chan_mask]
    flagged = flagged[:, :, chan_mask]
    freqs = freqs[chan_mask]
    print(f"trimmed to {freqs[0]:.1f}-{freqs[-1]:.1f} MHz  ({chan_mask.sum()} channels)")

    n_baseline = amp.shape[1]

    waterfall, flag_mask = get_avg_waterfall(amp, flagged)

    # drop time steps where every channel is flagged
    valid_time = flag_mask.mean(axis=1) < 1.0
    waterfall = waterfall[valid_time]
    flag_mask = flag_mask[valid_time]
    amp = amp[valid_time]
    flagged = flagged[valid_time]
    print(f"dropped {(~valid_time).sum()} fully-flagged time bins, {valid_time.sum()} remaining")

    unflagged_clean = waterfall[flag_mask == 0]
    global_vmin = np.percentile(unflagged_clean, 5)
    global_vmax = np.percentile(unflagged_clean, 95)

    # --- Per-baseline waterfalls ---
    n_show = min(args.n_baselines, n_baseline)
    baseline_indices = np.linspace(0, n_baseline - 1, n_show, dtype=int)
    ncols = 2
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    for i, bl in enumerate(baseline_indices):
        wf, fm = get_baseline_waterfall(amp, flagged, int(bl))
        patch = wf[:PATCH_SIZE, :]
        pflags = fm[:PATCH_SIZE, :]
        unflagged_vals = patch[pflags == 0]
        vmin = np.percentile(unflagged_vals, 5) if len(unflagged_vals) > 10 else global_vmin
        vmax = np.percentile(unflagged_vals, 95) if len(unflagged_vals) > 10 else global_vmax
        ax = axes[i]
        ax.imshow(patch.T, aspect='auto', origin='lower',
                  extent=[0, patch.shape[0], freqs[0], freqs[-1]],
                  vmin=vmin, vmax=vmax, cmap='plasma')
        ax.imshow(green_overlay(pflags), aspect='auto', origin='lower',
                  extent=[0, patch.shape[0], freqs[0], freqs[-1]])
        ax.set_title(f"baseline {bl}", fontsize=9)
        ax.set_xlabel("Time bins", fontsize=8)
        if i % ncols == 0:
            ax.set_ylabel("Freq (MHz)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n_show:]:
        ax.set_visible(False)
    plt.suptitle("Real MeerKAT — per-baseline waterfalls (green = flagged)", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "patches.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved patches.png")

    # --- Amplitude distribution (unflagged only) ---
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
    ax.set_title("Real MeerKAT — amplitude distribution (unflagged pixels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_dist.png", dpi=120)
    plt.close()
    print("saved amplitude_dist.png")

    # --- Mean spectrum (log scale, flagged pixels excluded, RFI bands annotated) ---
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
    ax.set_title("Real MeerKAT — mean spectrum (known RFI bands annotated)")
    plt.tight_layout()
    plt.savefig(out_dir / "mean_spectrum.png", dpi=120)
    plt.close()
    print("saved mean_spectrum.png")

    # --- Full waterfall ---
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(waterfall.T, aspect='auto', origin='lower',
                   extent=[0, waterfall.shape[0], freqs[0], freqs[-1]],
                   vmin=global_vmin, vmax=global_vmax, cmap='plasma')
    ax.imshow(green_overlay(flag_mask), aspect='auto', origin='lower',
              extent=[0, waterfall.shape[0], freqs[0], freqs[-1]])
    ax.set_xlabel("Time bins")
    ax.set_ylabel("Freq (MHz)")
    ax.set_title("Real MeerKAT — full waterfall (green = flagged)")
    plt.colorbar(im, ax=ax, label="Amplitude (Jy)", pad=0.01)
    plt.tight_layout()
    plt.savefig(out_dir / "waterfall_full.png", dpi=100)
    plt.close()
    print("saved waterfall_full.png")

    print(f"\nstats (unflagged pixels):")
    print(f"  mean={unflagged_clean.mean():.4f}  std={unflagged_clean.std():.4f}")
    print(f"  p5={np.percentile(unflagged_clean,5):.4f}  p95={np.percentile(unflagged_clean,95):.4f} Jy")
    print(f"\nall plots -> {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--field', type=int, default=None)
    parser.add_argument('--max-time', type=int, default=9999)
    parser.add_argument('--n-baselines', type=int, default=8)
    parser.add_argument('--freq-min', type=float, default=900.0)
    parser.add_argument('--freq-max', type=float, default=1650.0)
    main(parser.parse_args())
