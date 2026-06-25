import argparse
import time

import h5py
import numpy as np
from casacore.tables import table
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def to512(arr, sz):
    return resize(arr, (sz, sz), order=1, mode='edge', anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    chan_lo = int(hf.attrs['chan_lo'])
    chan_hi = chan_lo + int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf
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
        sr = tlo * n_baseline + bl
        D = root.getcol('DATA', startrow=sr, nrow=nt, rowincr=n_baseline)[:, chan_lo:chan_hi, :]
        theta = np.angle(D.mean(axis=2)).astype(np.float32)
        preds[u] = np.stack([hf[amp_key][u].astype(np.float32),
                             to512(np.cos(theta), sz), to512(np.sin(theta), sz)], 0)
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
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    main(ap.parse_args())
