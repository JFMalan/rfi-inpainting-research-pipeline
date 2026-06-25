import argparse
import time

import h5py
import numpy as np
from casacore.tables import table, maketabdesc, makecoldesc
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def ensure_weight_spectrum(ms, chunk=50000):
    if 'WEIGHT_SPECTRUM' in ms.colnames():
        log('WEIGHT_SPECTRUM exists, reusing')
        return
    log('adding WEIGHT_SPECTRUM (init = WEIGHT broadcast across channels)')
    desc = ms.getcoldesc('DATA'); desc['valueType'] = 'float'
    dminfo = ms.getdminfo('DATA'); dminfo['NAME'] = 'WEIGHT_SPECTRUM'
    ms.addcols(maketabdesc(makecoldesc('WEIGHT_SPECTRUM', desc)), dminfo)
    nchan = ms.getcell('DATA', 0).shape[0]
    n = ms.nrows()
    for s in range(0, n, chunk):
        nr = min(chunk, n - s)
        w = ms.getcol('WEIGHT', startrow=s, nrow=nr)
        ms.putcol('WEIGHT_SPECTRUM', np.repeat(w[:, None, :], nchan, axis=1).astype(np.float32),
                  startrow=s, nrow=nr)
    log(f'initialised WEIGHT_SPECTRUM ({n} rows)')


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    hf = h5py.File(args.h5, 'r')
    chan_lo = int(hf.attrs['chan_lo'])
    chan_hi = chan_lo + int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf

    ms = table(args.ms, readonly=False, ack=False)
    times = ms.getcol('TIME')
    n_time = len(np.unique(times))
    n_baseline = ms.nrows() // n_time
    ensure_weight_spectrum(ms)
    cap = hf[hole_key].shape[0] if args.max_units is None else min(args.max_units, hf[hole_key].shape[0])
    log(f"set hole weight = {args.frac} x WEIGHT for {cap} units  n_baseline={n_baseline}")

    for u in range(cap):
        bl = int(hf['baseline_id'][u]); nt = int(hf['native_n_time'][u])
        tlo = int(hf['time_lo'][u]) if has_tlo else 0
        hole = resize(hf[hole_key][u].astype(np.float32), (nt, chan_hi - chan_lo),
                      order=0, mode='edge', preserve_range=True) > 0.5
        sr = tlo * n_baseline + bl
        w_row = ms.getcol('WEIGHT', startrow=sr, nrow=nt, rowincr=n_baseline)
        ws = ms.getcol('WEIGHT_SPECTRUM', startrow=sr, nrow=nt, rowincr=n_baseline)
        wsb = ws[:, chan_lo:chan_hi, :]
        for p in range(ws.shape[2]):
            wsb[:, :, p] = np.where(hole, args.frac * w_row[:, p][:, None], wsb[:, :, p])
        ms.putcol('WEIGHT_SPECTRUM', ws, startrow=sr, nrow=nt, rowincr=n_baseline)

        fl = ms.getcol('FLAG', startrow=sr, nrow=nt, rowincr=n_baseline)
        fb = fl[:, chan_lo:chan_hi, :]
        for p in range(fl.shape[2]):
            fb[:, :, p] = np.where(hole, False, fb[:, :, p])
        ms.putcol('FLAG', fl, startrow=sr, nrow=nt, rowincr=n_baseline)
        if u == 0 or (u + 1) % 200 == 0:
            log(f"  {u + 1}/{cap}")

    ms.flush(); ms.close(); hf.close()
    log(f"done: hole weight = {args.frac} x WEIGHT, FLAG cleared in holes ({cap} units)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--frac', type=float, required=True)
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    main(ap.parse_args())
