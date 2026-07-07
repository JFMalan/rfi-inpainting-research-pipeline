import argparse
import time
from collections import defaultdict

import h5py
import numpy as np
from casacore.tables import table, maketabdesc, makecoldesc
from skimage.transform import resize

from classical_fill import dpss_basis, dpss_fill

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def ensure_column(root, out_col, src_col, chunk=50000):
    if out_col in root.colnames():
        log(f"column {out_col} exists, reusing")
        return
    log(f"adding {out_col} (copy of {src_col})")
    cd = makecoldesc(out_col, root.getcoldesc(src_col))
    dminfo = root.getdminfo(src_col); dminfo['NAME'] = out_col
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
    N = int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf; has_flo = 'freq_lo' in hf
    cap = hf[hole_key].shape[0]
    bl_arr  = hf['baseline_id'][:cap].astype(int)
    nt_arr  = hf['native_n_time'][:cap].astype(int)
    nc_arr  = hf['native_n_chan'][:cap].astype(int)
    tlo_arr = hf['time_lo'][:cap].astype(int) if has_tlo else np.zeros(cap, int)
    flo_arr = hf['freq_lo'][:cap].astype(int) if has_flo else np.zeros(cap, int)
    runs = defaultdict(list)
    for u in range(cap):
        runs[(tlo_arr[u], nt_arr[u])].append(u)

    root = table(args.ms, readonly=False, ack=False)
    n_time = len(np.unique(root.getcol('TIME')))
    n_baseline = root.nrows() // n_time
    ensure_column(root, args.out_col, 'DATA')
    npol = int(root.getcell('DATA', 0).shape[1])
    blc = [chan_lo, 0]; trc = [chan_lo + N - 1, npol - 1]
    A = dpss_basis(N, args.dpss_hw)
    log(f"DPSS-fill into {args.out_col}  runs={len(runs)}  n_baseline={n_baseline}  npol={npol}  "
        f"band={N}  hw={args.dpss_hw} K={A.shape[0]} lam={args.dpss_lam}")

    for (tlo, nt), units in runs.items():
        r0 = tlo * n_baseline; nrows = nt * n_baseline
        hole = np.zeros((nt, n_baseline, N), bool)
        for u in units:
            nc = nc_arr[u]; flo = flo_arr[u]; bl = bl_arr[u]
            h = resize(hf[hole_key][u].astype(np.float32), (nt, nc), order=0,
                       mode='edge', preserve_range=True) > 0.5
            hole[:, bl, flo:flo + nc] |= h
        d = root.getcolslice(args.out_col, blc, trc, [], r0, nrows).reshape(nt, n_baseline, N, npol)
        g = hole.reshape(nt * n_baseline, N)                    # identical gap pattern across baselines -> one solve
        for p in range(npol):
            V = d[..., p].reshape(nt * n_baseline, N).astype(np.complex128)
            d[..., p] = dpss_fill(V, g, A, args.dpss_lam).reshape(nt, n_baseline, N).astype(d.dtype)
        root.putcolslice(args.out_col, d.reshape(nrows, N, npol), blc, trc, [], r0, nrows)
        if args.unflag:
            fl = root.getcolslice('FLAG', blc, trc, [], r0, nrows).reshape(nt, n_baseline, N, npol)
            for p in range(npol):
                fl[..., p][hole] = False
            root.putcolslice('FLAG', fl.reshape(nrows, N, npol), blc, trc, [], r0, nrows)
        log(f"  run tlo={tlo} nt={nt}: {len(units)} tiles, {int(hole.sum())} hole cells/pol DPSS-filled")

    root.flush(); root.close(); hf.close()
    log(f"done: DPSS-filled {len(runs)} runs -> {args.out_col}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out-col', default='DPSSFILL_DATA', dest='out_col')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--unflag', action='store_true')
    ap.add_argument('--dpss-hw', type=float, default=0.1, dest='dpss_hw')
    ap.add_argument('--dpss-lam', type=float, default=0.1, dest='dpss_lam')
    main(ap.parse_args())
