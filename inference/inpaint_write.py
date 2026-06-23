import argparse
import time

import h5py
import numpy as np
from casacore.tables import table, maketabdesc, makecoldesc
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def resize_native(arr, n_time, n_chan, order=1, aa=False):
    return resize(arr, (n_time, n_chan), order=order, mode='edge',
                  anti_aliasing=aa, preserve_range=True).astype(np.float32)


def ensure_column(root, out_col, src_col, chunk=50000):
    if out_col in root.colnames():
        log(f"column {out_col} exists, reusing")
        return
    log(f"adding {out_col} (copy of {src_col})")
    cd = makecoldesc(out_col, root.getcoldesc(src_col))
    dminfo = root.getdminfo(src_col)
    dminfo['NAME'] = out_col          # unique data-manager name, else clashes with src_col's
    root.addcols(maketabdesc(cd), dminfo)
    n = root.nrows()
    for s in range(0, n, chunk):
        nr = min(chunk, n - s)
        root.putcol(out_col, root.getcol(src_col, startrow=s, nrow=nr), startrow=s, nrow=nr)
    log(f"initialised {out_col} ({n} rows)")


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    preds = np.load(args.preds)['preds']    # (cap, 3, sz, sz)
    cap = preds.shape[0]
    log(f"preds {preds.shape} from {args.preds}")

    hf = h5py.File(args.h5, 'r')
    chan_lo = int(hf.attrs['chan_lo'])
    full_n_chan = int(hf.attrs['full_n_chan'])
    chan_hi = chan_lo + full_n_chan
    has_tlo = 'time_lo' in hf

    root = table(args.ms, readonly=False, ack=False)
    src_col = args.src_col if args.src_col in root.colnames() else 'DATA'
    times = root.getcol('TIME')
    n_row = root.nrows()
    n_time = len(np.unique(times))
    if n_row % n_time != 0:
        raise RuntimeError(f"n_row {n_row} not divisible by n_time {n_time} — row map unsafe")
    n_baseline = n_row // n_time
    tb = times[:n_time * n_baseline].reshape(n_time, n_baseline)
    if not np.all(tb.max(axis=1) == tb.min(axis=1)):
        raise RuntimeError("MS rows not time-major — row map unsafe")
    log(f"MS {n_row} rows  n_time={n_time}  n_baseline={n_baseline}  src_col={src_col}")

    ensure_column(root, args.out_col, src_col)
    if args.field is not None:
        ms = root.query(f"FIELD_ID == {args.field}")
        st = ms.getcol('TIME'); n_row = ms.nrows(); n_time = len(np.unique(st))
        n_baseline = n_row // n_time
        log(f"field {args.field}: n_time={n_time}  n_baseline={n_baseline}")
    else:
        ms = root

    written = 0
    for u in range(cap):
        hole = hf[hole_key][u].astype(np.float32)
        divisor = hf['dn_divisor'][u].astype(np.float32)
        bl = int(hf['baseline_id'][u]); nt = int(hf['native_n_time'][u]); nc = int(hf['native_n_chan'][u])
        tlo = int(hf['time_lo'][u]) if has_tlo else 0
        if nc != full_n_chan:
            raise RuntimeError(f"unit {u}: native_n_chan {nc} != full_n_chan {full_n_chan}")
        amp_n = resize_native(preds[u, 0], nt, nc)
        div_n = resize_native(divisor, nt, nc)
        cos_n = resize_native(preds[u, 1], nt, nc)
        sin_n = resize_native(preds[u, 2], nt, nc)
        hole_n = resize_native(hole, nt, nc) > 0.5
        V = (amp_n * div_n * np.exp(1j * np.arctan2(sin_n, cos_n))).astype(np.complex64)

        sr = tlo * n_baseline + bl
        d = ms.getcol(args.out_col, startrow=sr, nrow=nt, rowincr=n_baseline)  # (nt, nchan_tot, npol)
        npol = d.shape[2]
        band = d[:, chan_lo:chan_hi, :]
        for p in range(npol):
            band[:, :, p] = np.where(hole_n, V, band[:, :, p])
        ms.putcol(args.out_col, d, startrow=sr, nrow=nt, rowincr=n_baseline)

        if args.unflag:
            fl = ms.getcol('FLAG', startrow=sr, nrow=nt, rowincr=n_baseline)
            fb = fl[:, chan_lo:chan_hi, :]
            for p in range(npol):
                fb[:, :, p] = np.where(hole_n, False, fb[:, :, p])
            ms.putcol('FLAG', fl, startrow=sr, nrow=nt, rowincr=n_baseline)

        written += 1
        if written == 1 or written % 100 == 0:
            log(f"  wrote {written}/{cap}  bl={bl} tlo={tlo} holes={int(hole_n.sum())}")

    ms.flush(); root.flush(); hf.close()
    log(f"done: {written} units -> {args.out_col}{' (FLAG cleared in holes)' if args.unflag else ''}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--out-col', default='INPAINTED_DATA', dest='out_col')
    ap.add_argument('--src-col', default='DATA', dest='src_col')
    ap.add_argument('--field', type=int, default=None)
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--unflag', action='store_true')
    main(ap.parse_args())
