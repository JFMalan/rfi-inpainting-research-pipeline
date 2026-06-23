import argparse
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from casacore.tables import table
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def amp_waterfall(d, chan_lo, chan_hi):
    return np.abs(d[:, chan_lo:chan_hi, :]).mean(axis=2)   # (nt, n_chan), pol-averaged


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    hf = h5py.File(args.h5, 'r')
    chan_lo = int(hf.attrs['chan_lo'])
    chan_hi = chan_lo + int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf

    ms = table(args.ms, readonly=True, ack=False)
    if args.inp_col not in ms.colnames():
        raise RuntimeError(f"{args.inp_col} not in MS — run the write-back first ({ms.colnames()})")
    times = ms.getcol('TIME')
    n_row = ms.nrows()
    n_time = len(np.unique(times))
    n_baseline = n_row // n_time
    log(f"MS {n_row} rows  n_time={n_time}  n_baseline={n_baseline}  cols: {args.src_col},{args.inp_col}")

    n = min(args.n, hf[hole_key].shape[0])
    fig, ax = plt.subplots(n, 4, figsize=(16, 3.2 * n))
    if n == 1:
        ax = ax[None, :]
    for r in range(n):
        bl = int(hf['baseline_id'][r]); nt = int(hf['native_n_time'][r])
        tlo = int(hf['time_lo'][r]) if has_tlo else 0
        sr = tlo * n_baseline + bl
        d_src = ms.getcol(args.src_col, startrow=sr, nrow=nt, rowincr=n_baseline)
        d_inp = ms.getcol(args.inp_col, startrow=sr, nrow=nt, rowincr=n_baseline)
        a_src = amp_waterfall(d_src, chan_lo, chan_hi)
        a_inp = amp_waterfall(d_inp, chan_lo, chan_hi)
        hole = resize(hf[hole_key][r].astype(np.float32), a_src.shape, order=0,
                      mode='edge', preserve_range=True) > 0.5
        diff = a_inp - a_src

        unmasked = a_src[~hole] if hole.any() else a_src
        vmin, vmax = np.percentile(unmasked, [1, 99])
        dlim = np.percentile(np.abs(diff), 99) + 1e-9
        panels = [(a_src, f"{args.src_col} (truth on sim)", 'inferno', vmin, vmax),
                  (hole.astype(float), 'inpainted region (holes)', 'gray', 0, 1),
                  (a_inp, f"{args.inp_col}", 'inferno', vmin, vmax),
                  (diff, 'inpainted - src (=0 outside holes)', 'coolwarm', -dlim, dlim)]
        for c, (img, title, cmap, lo, hi) in enumerate(panels):
            ax[r, c].imshow(img, aspect='auto', origin='lower', cmap=cmap, vmin=lo, vmax=hi)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(title, fontsize=9)
            if c == 0:
                ax[r, c].set_ylabel(f"bl {bl}", fontsize=8)
        log(f"  baseline {bl}: holes={int(hole.sum())}  "
            f"hole-MAE={np.abs(diff[hole]).mean() if hole.any() else 0:.4f}  "
            f"outside-max|diff|={np.abs(diff[~hole]).max() if (~hole).any() else 0:.2e}")

    fig.suptitle(f"MS write-back — amplitude (freq x time), {n} baselines", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    log(f"saved -> {args.out}")
    ms.close(); hf.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--src-col', default='DATA', dest='src_col')
    ap.add_argument('--inp-col', default='INPAINTED_DATA', dest='inp_col')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--n', type=int, default=6)
    main(ap.parse_args())
