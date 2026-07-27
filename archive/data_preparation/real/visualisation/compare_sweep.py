import argparse
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from casacore.tables import table
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'data_preparation' / 'real'))
from rfi_bands import LBAND_PERSISTENT_MHZ

RFI_BANDS = [(930, 960), (1170, 1300), (1525, 1630)]


def load_mean_spectrum(ms_path, freq_min, freq_max, field=None, max_time=512):
    ms = table(ms_path, readonly=True)
    if field is not None:
        ms = ms.query(f"FIELD_ID == {field}")
    cols = ms.colnames()
    col = 'CORRECTED_DATA' if 'CORRECTED_DATA' in cols else 'DATA'
    try:
        ms.getcell(col, 0)
    except Exception:
        col = 'DATA'
    data  = ms.getcol(col)
    flags = ms.getcol('FLAG')
    times = ms.getcol('TIME')
    ms.close()

    freqs_tab = table(ms_path + '/SPECTRAL_WINDOW')
    freqs = freqs_tab.getcol('CHAN_FREQ')[0] / 1e6
    freqs_tab.close()

    chan_mask = (freqs >= freq_min) & (freqs <= freq_max)
    data  = data[:, chan_mask, :]
    flags = flags[:, chan_mask, :]
    freqs = freqs[chan_mask]

    for flo, fhi in LBAND_PERSISTENT_MHZ:
        flags[:, (freqs >= flo) & (freqs <= fhi), :] = True

    amp     = np.abs(data).mean(axis=2).astype(np.float32)
    flagged = flags.any(axis=2)

    unique_times = np.unique(times)
    n_time     = min(len(unique_times), max_time)
    n_baseline = amp.shape[0] // len(unique_times)
    amp     = amp[:n_time * n_baseline].reshape(n_time, n_baseline, amp.shape[1])
    flagged = flagged[:n_time * n_baseline].reshape(n_time, n_baseline, flagged.shape[1])

    wf = np.ma.array(amp, mask=flagged).mean(axis=1).filled(np.nan)
    with np.errstate(all='ignore'):
        mean_spec = np.nanmean(wf, axis=0)
    return freqs, mean_spec


def load_patch_stats(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        flags = hf['flags'][:]
        data  = hf['data'][:]
        n_patches = int(hf.attrs.get('n_patches', flags.shape[0]))
    flag_fracs  = flags.mean(axis=(1, 2))
    unflagged   = data[flags == 0]
    return {
        'n_patches':    n_patches,
        'mean_ff':      float(flag_fracs.mean()),
        'median_ff':    float(np.median(flag_fracs)),
        'amp_mean':     float(unflagged.mean()) if len(unflagged) else 0.0,
        'amp_std':      float(unflagged.std())  if len(unflagged) else 0.0,
        'flag_fracs':   flag_fracs,
    }


def main(args):
    configs = json.loads(Path(args.configs).read_text())
    sweep_dir = Path(args.sweep_dir)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs, base_spec = load_mean_spectrum(
        args.ms, args.freq_min, args.freq_max, args.field
    )

    results = []
    for cfg in configs:
        h5 = sweep_dir / f"{cfg['id']}.h5"
        if not h5.exists():
            print(f"missing {h5}, skipping")
            continue
        stats = load_patch_stats(h5)
        results.append({'cfg': cfg, 'stats': stats})
        print(f"{cfg['id']:12s}  n={stats['n_patches']:5d}  "
              f"mean_ff={stats['mean_ff']:.3f}  amp_mean={stats['amp_mean']:.4f}")

    if not results:
        raise RuntimeError("no sweep outputs found in " + str(sweep_dir))

    # --- mean spectrum comparison ---
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(freqs, base_spec, linewidth=0.6, color='black', label='raw (no extra flagging)', zorder=10)
    cmap = plt.cm.tab10
    for i, r in enumerate(results):
        cfg = r['cfg']
        label = (f"{cfg['id']}  sc={cfg['sigma_clip']}  "
                 f"sm={cfg['smooth_bins']}  ff<={cfg['max_flag_frac']}")
        h5 = sweep_dir / f"{cfg['id']}.h5"
        with h5py.File(h5, 'r') as hf:
            if 'data' not in hf:
                continue
        ax.plot(freqs, base_spec, linewidth=0, alpha=0)
    ax.set_yscale('log')
    ax.set_xlabel("Freq (MHz)")
    ax.set_ylabel("Mean amplitude (Jy, log)")
    ax.set_title("Mean spectrum — sweep configs vs raw")
    for lo, hi in RFI_BANDS:
        ax.axvspan(lo, hi, color='red', alpha=0.1, linewidth=0)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_spectra.png", dpi=130)
    plt.close()

    # --- flag fraction distribution per config ---
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cfg   = r['cfg']
        stats = r['stats']
        ax.hist(stats['flag_fracs'], bins=40, color='steelblue', alpha=0.8, edgecolor='none')
        ax.axvline(stats['mean_ff'], color='red', linewidth=1,
                   label=f"mean={stats['mean_ff']:.3f}")
        ax.set_title(f"{cfg['id']}\nsc={cfg['sigma_clip']} sm={cfg['smooth_bins']}", fontsize=8)
        ax.set_xlabel("Flag frac / patch", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Count")
    plt.suptitle("Per-patch flag fraction distribution — sweep configs", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_flag_dist.png", dpi=130, bbox_inches="tight")
    plt.close()

    # --- summary table ---
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 4))
    ax.axis('off')
    headers = ['id', 'smooth_bins', 'sigma_clip', 'max_flag_frac',
               'n_patches', 'mean_ff', 'median_ff', 'amp_mean', 'amp_std']
    rows = []
    for r in results:
        cfg   = r['cfg']
        stats = r['stats']
        rows.append([
            cfg['id'], cfg['smooth_bins'], cfg['sigma_clip'], cfg['max_flag_frac'],
            stats['n_patches'],
            f"{stats['mean_ff']:.3f}",
            f"{stats['median_ff']:.3f}",
            f"{stats['amp_mean']:.4f}",
            f"{stats['amp_std']:.4f}",
        ])
    tbl = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_summary_table.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(f"plots saved -> {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms',         required=True)
    parser.add_argument('--configs',    required=True)
    parser.add_argument('--sweep-dir',  required=True)
    parser.add_argument('--output',     required=True)
    parser.add_argument('--field',      type=int,   default=None)
    parser.add_argument('--freq-min',   type=float, default=900.0)
    parser.add_argument('--freq-max',   type=float, default=1650.0)
    main(parser.parse_args())
