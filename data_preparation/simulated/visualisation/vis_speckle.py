import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load(path, n):
    with h5py.File(path, 'r') as f:
        key = 'clean' if 'clean' in f else 'data'
        fkey = 'mask' if 'mask' in f else 'flags'
        tot = f[key].shape[0]
        idx = np.linspace(0, tot - 1, min(n, tot)).astype(int)
        amp = f[key][idx].astype(np.float32)
        flags = f[fkey][idx].astype(np.float32)
        smooth = f['clean_smooth'][idx].astype(np.float32) if 'clean_smooth' in f else None
        fmin = float(f.attrs['freq_min_mhz']); fmax = float(f.attrs['freq_max_mhz'])
    return amp, flags, smooth, fmin, fmax


def main(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    sim_amp, sim_mask, sim_smooth, fmin, fmax = load(args.sim, args.n)
    real_amp, real_flags, _, _, _ = load(args.real, args.n)

    real_valid = real_flags < 0.5
    vmax = np.percentile(real_amp[real_valid], 99)

    ncol = 4
    fig, ax = plt.subplots(3, ncol, figsize=(4 * ncol, 11))
    for j in range(ncol):
        si = j * (sim_amp.shape[0] // ncol)
        ri = j * (real_amp.shape[0] // ncol)
        sm = np.ma.array(sim_amp[si], mask=sim_mask[si] > 0.5)
        rm = np.ma.array(real_amp[ri], mask=~real_valid[ri])
        ax[0, j].imshow(sm.T, aspect='auto', origin='lower', vmin=0, vmax=vmax,
                        cmap='plasma', extent=[0, 512, fmin, fmax])
        ax[0, j].set_title(f'sim speckle #{si}')
        if sim_smooth is not None:
            ax[1, j].imshow(sim_smooth[si].T, aspect='auto', origin='lower', vmin=0, vmax=vmax,
                            cmap='plasma', extent=[0, 512, fmin, fmax])
            ax[1, j].set_title(f'sim smooth (recoverable) #{si}')
        ax[2, j].imshow(rm.T, aspect='auto', origin='lower', vmin=0, vmax=vmax,
                        cmap='plasma', extent=[0, 512, fmin, fmax])
        ax[2, j].set_title(f'real #{ri}')
    ax[0, 0].set_ylabel('freq MHz'); ax[1, 0].set_ylabel('freq MHz'); ax[2, 0].set_ylabel('freq MHz')
    plt.tight_layout(); plt.savefig(out / 'speckle_vs_real.png', dpi=110); plt.close()

    # zoomed crop of an unflagged region: does the grain match?
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    sc = sim_amp[0][100:200, 100:200]
    rc = real_amp[0]
    rv = real_valid[0]
    rr, cc = np.where(rv)
    if rr.size:
        r0, c0 = int(np.median(rr)) - 50, int(np.median(cc)) - 50
        r0 = max(0, min(r0, 412)); c0 = max(0, min(c0, 412))
        rc = rc[r0:r0 + 100, c0:c0 + 100]
    ax[0].imshow(sc, cmap='plasma', vmin=0, vmax=vmax); ax[0].set_title('sim speckle (100x100 crop)')
    ax[1].imshow(rc, cmap='plasma', vmin=0, vmax=vmax); ax[1].set_title('real (100x100 crop)')
    plt.tight_layout(); plt.savefig(out / 'speckle_grain_crop.png', dpi=120); plt.close()

    print(f"saved -> {out}/speckle_vs_real.png and speckle_grain_crop.png", flush=True)
    print(f"sim speckle amp std (unflagged-equiv): {sim_amp[sim_mask<0.5].std():.4f}", flush=True)
    print(f"real amp std (unflagged):              {real_amp[real_valid].std():.4f}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', required=True)
    ap.add_argument('--real', default='/scratch3/users/jfmalan/rfi/real/variants/v1_upsample512.h5')
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=8)
    main(ap.parse_args())
