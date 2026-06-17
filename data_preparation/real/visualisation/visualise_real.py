import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from casacore.tables import table
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfi_bands import LBAND_PERSISTENT_MHZ

RFI_BANDS = [
    (930, 960),
    (1170, 1300),
    (1525, 1630),
]

PATCH_SIZE = 256


def load_ms(ms_path, max_time, field=None, column='DATA'):
    ms = table(ms_path, readonly=True)
    if field is not None:
        ms = ms.query(f"FIELD_ID == {field}")
    col = column if column in ms.colnames() else 'DATA'

    freqs_table = table(ms_path + '/SPECTRAL_WINDOW')
    freqs = freqs_table.getcol('CHAN_FREQ')[0] / 1e6
    freqs_table.close()
    persistent = np.zeros(len(freqs), bool)
    for flo, fhi in LBAND_PERSISTENT_MHZ:
        persistent |= (freqs >= flo) & (freqs <= fhi)

    times_all = ms.getcol('TIME')
    unique_times = np.unique(times_all)
    n_baseline = ms.nrows() // len(unique_times)
    n_time = min(len(unique_times), max_time)
    n_keep = n_time * n_baseline

    amp = np.empty((n_keep, len(freqs)), np.float32)
    flagged = np.empty((n_keep, len(freqs)), bool)
    block = n_baseline * 50
    for r0 in range(0, n_keep, block):
        nr = min(block, n_keep - r0)
        d = ms.getcol(col, r0, nr)
        fl = ms.getcol('FLAG', r0, nr)
        amp[r0:r0 + nr] = np.abs(d).mean(axis=2).astype(np.float32)
        f = fl.any(axis=2)
        f[:, persistent] = True
        flagged[r0:r0 + nr] = f
        print(f"  read rows {r0}/{n_keep}", flush=True)
    ms.close()

    n_chan = len(freqs)
    amp = amp.reshape(n_time, n_baseline, n_chan)
    flagged = flagged.reshape(n_time, n_baseline, n_chan)
    print(f"column: {col}  shape: ({n_time}, {n_baseline}, {n_chan})  "
          f"freq: {freqs[0]:.1f}-{freqs[-1]:.1f} MHz", flush=True)
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
    rgba[flag_mask_2d.T > 0] = [0.0, 1.0, 0.2, 0.85]
    return rgba


def plot_flagging_diagnostics(amp, flagged, freqs, avg_waterfall, avg_flag_mask, out_dir, patches_path=None):
    flag_per_freq    = flagged.mean(axis=(0, 1))
    flag_per_time    = flagged.mean(axis=(1, 2))
    flag_per_baseline = flagged.mean(axis=(0, 2))

    # --- main diagnostics: flag/freq, spectrum, flag/time ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    ax = axes[0]
    ax.plot(freqs, flag_per_freq, linewidth=0.7, color='steelblue')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Freq (MHz)")
    ax.set_ylabel("Flag fraction")
    ax.set_title("Flag fraction per channel (all baselines × time)")
    for lo, hi in RFI_BANDS:
        if lo < freqs[-1] and hi > freqs[0]:
            ax.axvspan(lo, hi, color='red', alpha=0.12, linewidth=0)
    ax.axhline(0.5, color='orange', linewidth=0.8, linestyle='--', label='50%')
    ax.axhline(0.9, color='red',    linewidth=0.8, linestyle='--', label='90%')
    ax.legend(fontsize=8)

    ax = axes[1]
    mean_unflagged = np.where(avg_flag_mask == 0, avg_waterfall, np.nan)
    mean_flagged   = np.where(avg_flag_mask  > 0, avg_waterfall, np.nan)
    with np.errstate(all='ignore'):
        mean_unflagged = np.nanmean(mean_unflagged, axis=0)
        mean_flagged   = np.nanmean(mean_flagged,   axis=0)
    ax.plot(freqs, mean_unflagged, linewidth=0.7, color='steelblue', label='unflagged mean')
    ax.plot(freqs, mean_flagged,   linewidth=0.7, color='red',       label='flagged mean', alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel("Freq (MHz)")
    ax.set_ylabel("Mean amplitude (Jy)")
    ax.set_title("Mean spectrum: flagged vs unflagged — spikes in unflagged = missed RFI")
    for lo, hi in RFI_BANDS:
        if lo < freqs[-1] and hi > freqs[0]:
            ax.axvspan(lo, hi, color='red', alpha=0.08, linewidth=0)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(flag_per_time, linewidth=0.8, color='steelblue')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Flag fraction")
    ax.set_title("Flag fraction per time bin — spikes = over-flagged timestamps")
    ax.axhline(0.5, color='orange', linewidth=0.8, linestyle='--', label='50%')
    ax.axhline(0.9, color='red',    linewidth=0.8, linestyle='--', label='90%')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "flag_diagnostics.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("saved flag_diagnostics.png")

    # --- zoomed spectrum panels around known missed RFI regions ---
    zoom_regions = [(1050, 1160), (1350, 1430), (1480, 1530)]
    fig, axes = plt.subplots(1, len(zoom_regions), figsize=(15, 4))
    for ax, (flo, fhi) in zip(axes, zoom_regions):
        mask = (freqs >= flo) & (freqs <= fhi)
        if mask.sum() < 2:
            ax.set_visible(False)
            continue
        ax.plot(freqs[mask], mean_unflagged[mask], linewidth=0.9, color='steelblue', label='unflagged')
        ax.plot(freqs[mask], mean_flagged[mask],   linewidth=0.9, color='red', alpha=0.7, label='flagged')
        ax.set_yscale('log')
        ax.set_xlabel("Freq (MHz)", fontsize=8)
        ax.set_ylabel("Mean amplitude (Jy)", fontsize=8)
        ax.set_title(f"{flo}–{fhi} MHz", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    plt.suptitle("Zoomed mean spectrum — missed RFI shows as spikes above local continuum", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "flag_diagnostics_zoom.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("saved flag_diagnostics_zoom.png")

    # --- flag fraction per baseline ---
    n_baseline = flag_per_baseline.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, n_baseline // 10), 4))
    ax.bar(np.arange(n_baseline), flag_per_baseline, width=1.0, color='steelblue', alpha=0.8)
    ax.set_xlabel("Baseline index")
    ax.set_ylabel("Flag fraction")
    ax.set_title("Flag fraction per baseline — outliers = bad antennas")
    ax.axhline(flag_per_baseline.mean(), color='red', linewidth=0.8, linestyle='--',
               label=f"mean={flag_per_baseline.mean():.3f}")
    ax.axhline(flag_per_baseline.mean() + 3 * flag_per_baseline.std(),
               color='orange', linewidth=0.8, linestyle='--', label='mean+3σ')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "flag_per_baseline.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("saved flag_per_baseline.png")

    # --- flag fraction CDF per channel ---
    sorted_ff = np.sort(flag_per_freq)
    cdf = np.arange(1, len(sorted_ff) + 1) / len(sorted_ff)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sorted_ff, cdf, linewidth=1.0, color='steelblue')
    ax.set_xlabel("Flag fraction")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of per-channel flag fraction")
    for thresh, label, col in [(0.1, '10%', 'green'), (0.5, '50%', 'orange'), (0.9, '90%', 'red')]:
        frac_above = (flag_per_freq > thresh).mean()
        ax.axvline(thresh, color=col, linewidth=0.8, linestyle='--',
                   label=f">{thresh:.0%}: {frac_above:.1%} of channels")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "flag_cdf.png", dpi=130)
    plt.close()
    print("saved flag_cdf.png")

    # --- stdout statistics ---
    top_freq_idx = np.argsort(flag_per_freq)[-10:][::-1]
    print("\ntop 10 most-flagged channels:")
    for idx in top_freq_idx:
        print(f"  {freqs[idx]:.1f} MHz  flag={flag_per_freq[idx]:.3f}")
    top_bl_idx = np.argsort(flag_per_baseline)[-10:][::-1]
    print("\ntop 10 most-flagged baselines:")
    for idx in top_bl_idx:
        print(f"  baseline {idx}  flag={flag_per_baseline[idx]:.3f}")
    print(f"\noverall flag fraction: {flagged.mean():.4f}")
    print(f"channels >10% flagged: {(flag_per_freq > 0.1).sum()}/{len(flag_per_freq)}")
    print(f"channels >50% flagged: {(flag_per_freq > 0.5).sum()}/{len(flag_per_freq)}")
    print(f"channels >90% flagged: {(flag_per_freq > 0.9).sum()}/{len(flag_per_freq)}")

    # --- per-patch flag fraction histogram ---
    if patches_path is not None:
        try:
            with h5py.File(patches_path, 'r') as hf:
                patch_flags = hf['flags'][:]
            patch_flag_fracs = patch_flags.mean(axis=(1, 2))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(patch_flag_fracs, bins=50, color='steelblue', alpha=0.8, edgecolor='none')
            ax.axvline(patch_flag_fracs.mean(), color='red', linewidth=1,
                       label=f"mean={patch_flag_fracs.mean():.3f}")
            ax.set_xlabel("Flag fraction per patch")
            ax.set_ylabel("Count")
            ax.set_title(f"Per-patch flag fraction distribution ({len(patch_flag_fracs)} patches)")
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / "patch_flag_hist.png", dpi=130)
            plt.close()
            print("saved patch_flag_hist.png")
            high = patch_flag_fracs[patch_flag_fracs > 0.5]
            print(f"  patches with >50% flags: {len(high)}/{len(patch_flag_fracs)} "
                  f"({100*len(high)/len(patch_flag_fracs):.1f}%)")
        except Exception as e:
            print(f"skipped patch_flag_hist: {e}")


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading MS...")
    amp, flagged, freqs = load_ms(args.ms, args.max_time, field=args.field, column=args.column)

    chan_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    amp = amp[:, :, chan_mask]
    flagged = flagged[:, :, chan_mask]
    freqs = freqs[chan_mask]
    print(f"trimmed to {freqs[0]:.1f}-{freqs[-1]:.1f} MHz  ({chan_mask.sum()} channels)")

    n_baseline = amp.shape[1]

    avg_waterfall, avg_flag_mask = get_avg_waterfall(amp, flagged)

    # drop time steps where every channel is flagged
    valid_time = avg_flag_mask.mean(axis=1) < 1.0
    avg_waterfall = avg_waterfall[valid_time]
    avg_flag_mask = avg_flag_mask[valid_time]
    amp = amp[valid_time]
    flagged = flagged[valid_time]
    print(f"dropped {(~valid_time).sum()} fully-flagged time bins, {valid_time.sum()} remaining")

    waterfall, flag_mask = avg_waterfall, avg_flag_mask

    unflagged_clean = waterfall[flag_mask == 0]
    global_vmin = np.percentile(unflagged_clean, 1)
    global_vmax = args.vmax if args.vmax is not None else np.percentile(unflagged_clean, 90)

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
        vmin = np.percentile(unflagged_vals, 2) if len(unflagged_vals) > 10 else global_vmin
        vmax = args.vmax if args.vmax is not None else (np.percentile(unflagged_vals, 90) if len(unflagged_vals) > 10 else global_vmax)
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

    # --- Amplitude distribution (unflagged only, all baselines) ---
    avg_unflagged = avg_waterfall[avg_flag_mask == 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.percentile(avg_unflagged, 0.5),
                       np.percentile(avg_unflagged, 99.5), 120)
    ax.hist(avg_unflagged, bins=bins, density=True, color='steelblue', alpha=0.8)
    ax.axvline(avg_unflagged.mean(), color='red', linewidth=1,
               label=f"mean={avg_unflagged.mean():.4f}")
    ax.axvline(np.median(avg_unflagged), color='orange', linewidth=1, linestyle='--',
               label=f"median={np.median(avg_unflagged):.4f}")
    ax.set_xlabel("Amplitude (Jy)")
    ax.set_ylabel("Density")
    ax.set_title("Real MeerKAT — amplitude distribution (unflagged pixels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_dist.png", dpi=120)
    plt.close()
    print("saved amplitude_dist.png")

    # --- Mean spectrum (log scale, flagged pixels excluded, RFI bands annotated) ---
    mean_spec = np.where(avg_flag_mask == 0, avg_waterfall, np.nan)
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

    print(f"\nstats (unflagged, baseline-averaged):")
    print(f"  mean={avg_unflagged.mean():.4f}  std={avg_unflagged.std():.4f}")
    print(f"  p5={np.percentile(avg_unflagged,5):.4f}  p90={np.percentile(avg_unflagged,90):.4f} Jy")

    plot_flagging_diagnostics(amp, flagged, freqs, avg_waterfall, avg_flag_mask, out_dir, args.patches)

    if args.patches:
        plot_patches_hdf5(args.patches, out_dir, args.n_patches_show)

    print(f"\nall plots -> {out_dir}/")


def plot_patches_hdf5(h5_path, out_dir, n_show, per_page=6):
    with h5py.File(h5_path, 'r') as hf:
        n_total   = hf['data'].shape[0]
        indices   = np.linspace(0, n_total - 1, min(n_show, n_total), dtype=int)
        patches   = hf['data'][indices]
        flags     = hf['flags'][indices]
        has_raw   = 'data_raw' in hf
        raw       = hf['data_raw'][indices] if has_raw else None
        if 'freq_min_patch' in hf:
            patch_fmin = hf['freq_min_patch'][:][indices]
            patch_fmax = hf['freq_max_patch'][:][indices]
        else:
            patch_fmin = np.full(len(indices), hf.attrs['freq_min_mhz'])
            patch_fmax = np.full(len(indices), hf.attrs['freq_max_mhz'])

    patch_dir = out_dir / "patches_hdf5"
    patch_dir.mkdir(exist_ok=True)

    # 2 columns per patch (raw | dn), 1 patch per row, 6 rows per page
    patches_per_row = 1
    ncols  = patches_per_row * 2
    pages  = (len(indices) + per_page - 1) // per_page
    n_saved = 0

    for page in range(pages):
        sl           = slice(page * per_page, (page + 1) * per_page)
        page_indices = indices[sl]
        page_patches = patches[sl]
        page_flags   = flags[sl]
        page_raw     = raw[sl] if has_raw else None
        page_fmin    = patch_fmin[sl]
        page_fmax    = patch_fmax[sl]
        n            = len(page_indices)
        nrows        = (n + patches_per_row - 1) // patches_per_row

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3.5))
        axes = np.array(axes).reshape(nrows, ncols)

        for i in range(n):
            row = i // patches_per_row
            col = (i % patches_per_row) * 2
            patch = page_patches[i]
            fm    = page_flags[i]
            fmin  = page_fmin[i]
            fmax  = page_fmax[i]

            unflagged_vals = patch[fm == 0]
            dn_vmin = np.percentile(unflagged_vals, 2)  if len(unflagged_vals) > 10 else patch.min()
            dn_vmax = np.percentile(unflagged_vals, 90) if len(unflagged_vals) > 10 else patch.max()

            # left: raw pre-DN
            ax_raw = axes[row, col]
            if page_raw is not None:
                r = page_raw[i]
                unfl_raw = r[fm == 0]
                rv_min = np.percentile(unfl_raw, 2)  if len(unfl_raw) > 10 else r.min()
                rv_max = np.percentile(unfl_raw, 90) if len(unfl_raw) > 10 else r.max()
                ax_raw.imshow(r.T, aspect='auto', origin='lower',
                              extent=[0, r.shape[0], fmin, fmax],
                              vmin=rv_min, vmax=rv_max, cmap='plasma')
                ax_raw.imshow(green_overlay(fm), aspect='auto', origin='lower',
                              extent=[0, r.shape[0], fmin, fmax])
                ax_raw.set_title(f"patch {page_indices[i]}  raw", fontsize=7)
            else:
                ax_raw.set_visible(False)
            ax_raw.tick_params(labelsize=5)
            if col == 0:
                ax_raw.set_ylabel("Freq (MHz)", fontsize=6)

            # right: post-DN with flags
            ax_dn = axes[row, col + 1]
            ax_dn.imshow(patch.T, aspect='auto', origin='lower',
                         extent=[0, patch.shape[0], fmin, fmax],
                         vmin=dn_vmin, vmax=dn_vmax, cmap='plasma')
            ax_dn.imshow(green_overlay(fm), aspect='auto', origin='lower',
                         extent=[0, patch.shape[0], fmin, fmax])
            ax_dn.set_title(f"patch {page_indices[i]}  dn  flag={fm.mean():.2f}", fontsize=7)
            ax_dn.tick_params(labelsize=5)

            if row == nrows - 1:
                ax_raw.set_xlabel("Time bins", fontsize=6)
                ax_dn.set_xlabel("Time bins", fontsize=6)

        for i in range(n, nrows * patches_per_row):
            row = i // patches_per_row
            col = (i % patches_per_row) * 2
            axes[row, col].set_visible(False)
            axes[row, col + 1].set_visible(False)

        plt.suptitle(
            f"Real MeerKAT — raw (left) vs post-DN (right), green = flagged  "
            f"[page {page + 1}/{pages}, {n_total} total]",
            y=1.01)
        plt.tight_layout()
        out_path = patch_dir / f"page_{page + 1:03d}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        n_saved += n

    print(f"saved {pages} pages -> {patch_dir}/  ({n_saved} of {n_total} patches shown)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms',             required=True)
    parser.add_argument('--output',         required=True)
    parser.add_argument('--patches',        default=None)
    parser.add_argument('--column',         default='DATA')
    parser.add_argument('--field',          type=int,   default=None)
    parser.add_argument('--max-time',       type=int,   default=9999)
    parser.add_argument('--n-baselines',    type=int,   default=16)
    parser.add_argument('--n-patches-show', type=int,   default=200)
    parser.add_argument('--freq-min',       type=float, default=900.0)
    parser.add_argument('--freq-max',       type=float, default=1650.0)
    parser.add_argument('--vmax',           type=float, default=None)
    main(parser.parse_args())
