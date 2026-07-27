import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from casacore.tables import table


def extract_waterfall(ms_path, max_time=512):
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
    freqs = freqs_table.getcol('CHAN_FREQ')[0]
    freqs_table.close()

    amp = np.abs(data).mean(axis=2).astype(np.float32)
    flagged = flags.any(axis=2)

    unique_times = np.unique(times)
    n_time = min(len(unique_times), max_time)
    n_chan = amp.shape[1]
    n_baseline = amp.shape[0] // len(unique_times)

    amp = amp[:n_time * n_baseline]
    flagged = flagged[:n_time * n_baseline]

    amp_3d = np.ma.array(
        amp.reshape(n_time, n_baseline, n_chan),
        mask=flagged.reshape(n_time, n_baseline, n_chan)
    )
    waterfall = amp_3d.mean(axis=1).filled(0.0).astype(np.float32)
    all_flagged = flagged.reshape(n_time, n_baseline, n_chan).all(axis=1)

    return waterfall, all_flagged.astype(np.float32), freqs / 1e6


def main(args):
    print("extracting simulated waterfall...")
    sim = np.load(args.sim_waterfall)
    sim_meta = np.load(args.sim_waterfall.replace('.npy', '.meta.npy'))
    sim_freqs = np.linspace(sim_meta[0], sim_meta[1], sim.shape[1])

    print("extracting real waterfall...")
    real, real_flags, real_freqs = extract_waterfall(args.real_ms, max_time=args.max_time)

    # trim both to same number of channels for comparison (use min)
    n_chan = min(sim.shape[1], real.shape[1])
    sim = sim[:, :n_chan]
    sim_freqs = sim_freqs[:n_chan]
    real = real[:, :n_chan]
    real_freqs = real_freqs[:n_chan]

    sim_flat = sim.flatten()
    real_unmasked = real[real_flags[:, :n_chan] == 0].flatten() if real_flags is not None else real.flatten()

    print(f"\nsimulated  : mean={sim_flat.mean():.4f}  std={sim_flat.std():.4f}  "
          f"p5={np.percentile(sim_flat,5):.4f}  p95={np.percentile(sim_flat,95):.4f} Jy")
    print(f"real       : mean={real_unmasked.mean():.4f}  std={real_unmasked.std():.4f}  "
          f"p5={np.percentile(real_unmasked,5):.4f}  p95={np.percentile(real_unmasked,95):.4f} Jy")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Waterfall comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_show = min(256, sim.shape[0], real.shape[0])
    vmin = np.percentile(np.concatenate([sim[:n_show].flatten(), real_unmasked]), 1)
    vmax = np.percentile(np.concatenate([sim[:n_show].flatten(), real_unmasked]), 99)

    axes[0].imshow(sim[:n_show].T, aspect='auto', origin='lower',
                   extent=[0, n_show, sim_freqs[0], sim_freqs[-1]],
                   vmin=vmin, vmax=vmax, cmap='plasma')
    axes[0].set_title('simulated')
    axes[0].set_xlabel('Time bins')
    axes[0].set_ylabel('Freq (MHz)')

    axes[1].imshow(real[:n_show].T, aspect='auto', origin='lower',
                   extent=[0, n_show, real_freqs[0], real_freqs[-1]],
                   vmin=vmin, vmax=vmax, cmap='plasma')
    axes[1].set_title('real MeerKAT')
    axes[1].set_xlabel('Time bins')

    plt.tight_layout()
    plt.savefig(out_dir / 'compare_waterfalls.png', dpi=120)
    plt.close()

    # --- Amplitude distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(np.concatenate([sim_flat, real_unmasked]), 99.5), 100)
    ax.hist(sim_flat, bins=bins, density=True, alpha=0.6, label='simulated')
    ax.hist(real_unmasked, bins=bins, density=True, alpha=0.6, label='real (unflagged)')
    ax.set_xlabel('Amplitude (Jy)')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'compare_amplitude_dist.png', dpi=120)
    plt.close()

    # --- Mean spectrum ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sim_freqs, sim[:n_show].mean(axis=0), label='simulated', linewidth=0.8)
    ax.plot(real_freqs, real[:n_show].mean(axis=0), label='real', linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Freq (MHz)')
    ax.set_ylabel('Mean amplitude (Jy)')
    ax.set_title('Mean spectrum')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'compare_spectra.png', dpi=120)
    plt.close()

    print(f"\nplots saved to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-waterfall', required=True, help='path to waterfall.npy from simulate.sh')
    parser.add_argument('--real-ms', required=True, help='path to real MeerKAT MS')
    parser.add_argument('--output', required=True, help='output directory for plots')
    parser.add_argument('--max-time', type=int, default=512, help='max time bins to load from real MS')
    main(parser.parse_args())
