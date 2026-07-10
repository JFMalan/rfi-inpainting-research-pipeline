import argparse
import time

import h5py
import numpy as np
from casacore.tables import table
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def to512(arr, sz):
    return resize(arr, (sz, sz), order=1, mode='edge', anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


def smooth_component(amp, mask, sigma=1.0):
    # mirror data.py: freq-interp across the hole, then 2D low-pass. This is the BEST a perfect
    # decompose model could output in the holes (smooth recoverable amplitude, no fabricated grain).
    filled = amp.copy()
    idx = np.arange(amp.shape[1])
    for t in range(amp.shape[0]):
        row = filled[t]; keep = mask[t] < 0.5
        if keep.sum() < 4:
            filled[t] = row.mean() if keep.any() else 1.0
            continue
        filled[t] = np.interp(idx, idx[keep], row[keep])
    return gaussian_filter(filled, sigma=sigma, mode='nearest').astype(np.float32)


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    chan_lo = int(hf.attrs['chan_lo'])
    has_tlo = 'time_lo' in hf
    has_flo = 'freq_lo' in hf
    amp_key = 'clean' if args.sim else 'data'

    root = table(args.ms, readonly=True, ack=False)
    times = root.getcol('TIME')
    n_time = len(np.unique(times))
    n_baseline = root.nrows() // n_time

    n_units = hf[hole_key].shape[0]
    cap = n_units if args.max_units is None else min(args.max_units, n_units)
    log(f"phase-fix oracle preds for {cap}/{n_units} units  n_baseline={n_baseline}")

    preds = np.empty((cap, 3, sz, sz), dtype=np.float32)
    for u in range(cap):
        bl = int(hf['baseline_id'][u]); nt = int(hf['native_n_time'][u]); nc = int(hf['native_n_chan'][u])
        tlo = int(hf['time_lo'][u]) if has_tlo else 0
        flo = int(hf['freq_lo'][u]) if has_flo else 0
        clo = chan_lo + flo; chi = clo + nc
        sr = tlo * n_baseline + bl
        D = root.getcol('DATA', startrow=sr, nrow=nt, rowincr=n_baseline)[:, clo:chi, :]
        theta = np.angle(D.mean(axis=2)).astype(np.float32)
        amp = hf[amp_key][u].astype(np.float32)
        if args.smooth_amp:
            amp = smooth_component(amp, hf[hole_key][u].astype(np.float32), args.smooth_sigma)
        preds[u] = np.stack([amp, to512(np.cos(theta), sz), to512(np.sin(theta), sz)], 0)
        if u == 0 or (u + 1) % 100 == 0:
            log(f"  built {u + 1}/{cap}  bl={bl} tlo={tlo}")

    root.close(); hf.close()
    np.savez(args.out_preds, preds=preds)
    log(f"saved {cap} phase-fix oracle preds {preds.shape} -> {args.out_preds}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out-preds', required=True, dest='out_preds')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--smooth-amp', action='store_true', dest='smooth_amp',
                    help='write smooth_component(clean) amplitude = the decompose-model ceiling')
    ap.add_argument('--smooth-sigma', type=float, default=1.0, dest='smooth_sigma')
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    main(ap.parse_args())
