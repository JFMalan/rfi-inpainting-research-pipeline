import argparse
import time
from collections import defaultdict

import h5py
import numpy as np
from casacore.tables import table, maketabdesc, makecoldesc
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def ensure_column(root, out_col, src_col, chunk=50000):
    if out_col in root.colnames():
        log(f"column {out_col} exists, reusing")
        return
    log(f"adding {out_col} (copy of {src_col})")
    cd = makecoldesc(out_col, root.getcoldesc(src_col))
    dminfo = root.getdminfo(src_col)
    dminfo['NAME'] = out_col
    root.addcols(maketabdesc(cd), dminfo)
    n = root.nrows()
    for s in range(0, n, chunk):
        nr = min(chunk, n - s)
        root.putcol(out_col, root.getcol(src_col, startrow=s, nrow=nr), startrow=s, nrow=nr)
    log(f"initialised {out_col} ({n} rows)")


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    hf = h5py.File(args.h5, 'r')
    chan_lo = int(hf.attrs['chan_lo'])
    full_n_chan = int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf
    has_flo = 'freq_lo' in hf
    cap = hf[hole_key].shape[0]

    bl_arr  = hf['baseline_id'][:cap].astype(int)
    nt_arr  = hf['native_n_time'][:cap].astype(int)
    nc_arr  = hf['native_n_chan'][:cap].astype(int)
    tlo_arr = hf['time_lo'][:cap].astype(int) if has_tlo else np.zeros(cap, int)
    flo_arr = hf['freq_lo'][:cap].astype(int) if has_flo else np.zeros(cap, int)
    groups = defaultdict(list)
    for u in range(cap):
        groups[(bl_arr[u], tlo_arr[u])].append(u)

    root = table(args.ms, readonly=False, ack=False)
    times = root.getcol('TIME')
    n_time = len(np.unique(times))
    n_baseline = root.nrows() // n_time
    ensure_column(root, args.out_col, 'DATA')
    log(f"mean-fill into {args.out_col}  groups={len(groups)}  n_baseline={n_baseline}")

    written = 0
    for (bl, tlo), units in groups.items():
        nt = nt_arr[units[0]]
        sr = tlo * n_baseline + bl
        d = root.getcol(args.out_col, startrow=sr, nrow=nt, rowincr=n_baseline)
        npol = d.shape[2]
        band = d[:, chan_lo:chan_lo + full_n_chan, :]

        hole = np.zeros((nt, full_n_chan), bool)
        for u in units:
            nc = nc_arr[u]; flo = flo_arr[u]
            h = resize(hf[hole_key][u].astype(np.float32), (nt, nc), order=0,
                       mode='edge', preserve_range=True) > 0.5
            hole[:, flo:flo + nc] |= h

        good = (~hole)[:, :, None].astype(np.float32)            # (nt, N, 1)
        cnt = good.sum(axis=0)                                   # (N, npol-broadcast)
        meanc = (band * good).sum(axis=0) / np.maximum(cnt, 1.0)  # (N, npol) per-channel time-mean of unflagged
        fill = np.broadcast_to(meanc[None], band.shape)
        hole3 = hole[:, :, None]
        d[:, chan_lo:chan_lo + full_n_chan, :] = np.where(hole3, fill, band)
        root.putcol(args.out_col, d, startrow=sr, nrow=nt, rowincr=n_baseline)

        if args.unflag:
            fl = root.getcol('FLAG', startrow=sr, nrow=nt, rowincr=n_baseline)
            fb = fl[:, chan_lo:chan_lo + full_n_chan, :]
            fb[:] = np.where(hole3, False, fb)
            root.putcol('FLAG', fl, startrow=sr, nrow=nt, rowincr=n_baseline)

        written += 1
        if written == 1 or written % 100 == 0:
            log(f"  filled {written}/{len(groups)}  bl={bl} tlo={tlo} holes={int(hole.sum())}")

    root.flush(); root.close(); hf.close()
    log(f"done: {written} baselines mean-filled -> {args.out_col}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out-col', default='MEANFILL_DATA', dest='out_col')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--unflag', action='store_true')
    main(ap.parse_args())
