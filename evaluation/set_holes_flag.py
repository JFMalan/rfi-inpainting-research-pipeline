import argparse
import time

import h5py
import numpy as np
from casacore.tables import table
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    hf = h5py.File(args.h5, 'r')
    chan_lo = int(hf.attrs['chan_lo'])
    chan_hi = chan_lo + int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf
    set_val = args.mode == 'set'

    ms = table(args.ms, readonly=False, ack=False)
    times = ms.getcol('TIME')
    n_time = len(np.unique(times))
    n_baseline = ms.nrows() // n_time
    cap = hf[hole_key].shape[0] if args.max_units is None else min(args.max_units, hf[hole_key].shape[0])
    log(f"{args.mode} FLAG at holes for {cap} units  n_baseline={n_baseline}")

    for u in range(cap):
        bl = int(hf['baseline_id'][u]); nt = int(hf['native_n_time'][u])
        tlo = int(hf['time_lo'][u]) if has_tlo else 0
        hole = resize(hf[hole_key][u].astype(np.float32), (nt, chan_hi - chan_lo),
                      order=0, mode='edge', preserve_range=True) > 0.5
        sr = tlo * n_baseline + bl
        fl = ms.getcol('FLAG', startrow=sr, nrow=nt, rowincr=n_baseline)
        band = fl[:, chan_lo:chan_hi, :]
        for p in range(band.shape[2]):
            band[:, :, p] = np.where(hole, set_val, band[:, :, p])
        ms.putcol('FLAG', fl, startrow=sr, nrow=nt, rowincr=n_baseline)
        if u == 0 or (u + 1) % 200 == 0:
            log(f"  {u + 1}/{cap}")

    ms.flush(); ms.close(); hf.close()
    log(f"done: FLAG {args.mode} at {cap} units' holes")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--mode', choices=['set', 'clear'], required=True)
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    main(ap.parse_args())
