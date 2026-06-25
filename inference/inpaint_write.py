import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from casacore.tables import table, maketabdesc, makecoldesc
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'data_preparation'))
from tiling import feather_weight

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


def ensure_weight_spectrum(root, chunk=50000):
    if 'WEIGHT_SPECTRUM' in root.colnames():
        log('WEIGHT_SPECTRUM exists, reusing')
        return
    log('adding WEIGHT_SPECTRUM (init = WEIGHT broadcast across channels)')
    desc = root.getcoldesc('DATA'); desc['valueType'] = 'float'
    dminfo = root.getdminfo('DATA'); dminfo['NAME'] = 'WEIGHT_SPECTRUM'
    root.addcols(maketabdesc(makecoldesc('WEIGHT_SPECTRUM', desc)), dminfo)
    nchan = root.getcell('DATA', 0).shape[0]
    n = root.nrows()
    for s in range(0, n, chunk):
        nr = min(chunk, n - s)
        w = root.getcol('WEIGHT', startrow=s, nrow=nr)
        root.putcol('WEIGHT_SPECTRUM', np.repeat(w[:, None, :], nchan, axis=1).astype(np.float32),
                    startrow=s, nrow=nr)
    log(f'initialised WEIGHT_SPECTRUM ({n} rows)')


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    preds = np.load(args.preds)['preds']    # (cap, 3, sz, sz)
    cap = preds.shape[0]
    log(f"preds {preds.shape} from {args.preds}")

    hf = h5py.File(args.h5, 'r')
    chan_lo = int(hf.attrs['chan_lo'])
    full_n_chan = int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf
    has_flo = 'freq_lo' in hf

    bl_arr  = hf['baseline_id'][:cap].astype(int)
    nt_arr  = hf['native_n_time'][:cap].astype(int)
    nc_arr  = hf['native_n_chan'][:cap].astype(int)
    tlo_arr = hf['time_lo'][:cap].astype(int) if has_tlo else np.zeros(cap, int)
    flo_arr = hf['freq_lo'][:cap].astype(int) if has_flo else np.zeros(cap, int)
    starts = sorted(set(int(x) for x in flo_arr))
    tile_w = int(nc_arr[0])
    log(f"tiles: starts={starts} width={tile_w} band={full_n_chan}")

    groups = defaultdict(list)
    for u in range(cap):
        groups[(bl_arr[u], tlo_arr[u])].append(u)

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
    log(f"MS {n_row} rows  n_time={n_time}  n_baseline={n_baseline}  src_col={src_col}  groups={len(groups)}")

    ensure_column(root, args.out_col, src_col)
    if args.weight_frac is not None:
        ensure_weight_spectrum(root)
    if args.field is not None:
        ms = root.query(f"FIELD_ID == {args.field}")
        st = ms.getcol('TIME'); n_row = ms.nrows(); n_time = len(np.unique(st))
        n_baseline = n_row // n_time
        log(f"field {args.field}: n_time={n_time}  n_baseline={n_baseline}")
    else:
        ms = root

    written = 0
    for (bl, tlo), units in groups.items():
        nt = nt_arr[units[0]]
        sr = tlo * n_baseline + bl
        d = ms.getcol(args.out_col, startrow=sr, nrow=nt, rowincr=n_baseline)  # (nt, nchan_tot, npol)
        npol = d.shape[2]

        vnum = np.zeros((nt, full_n_chan), dtype=np.complex64)
        wsum = np.zeros((nt, full_n_chan), dtype=np.float32)
        hany = np.zeros((nt, full_n_chan), dtype=bool)
        for u in units:
            nc = nc_arr[u]; flo = flo_arr[u]
            amp_n = resize_native(preds[u, 0], nt, nc)
            div_n = resize_native(hf['dn_divisor'][u], nt, nc)
            cos_n = resize_native(preds[u, 1], nt, nc)
            sin_n = resize_native(preds[u, 2], nt, nc)
            V = (amp_n * div_n * np.exp(1j * np.arctan2(sin_n, cos_n))).astype(np.complex64)
            hole_n = resize_native(hf[hole_key][u].astype(np.float32), nt, nc) > 0.5
            w = feather_weight(flo, nc, starts, tile_w)[None, :].astype(np.float32) * hole_n
            vnum[:, flo:flo + nc] += w * V
            wsum[:, flo:flo + nc] += w
            hany[:, flo:flo + nc] |= hole_n
        vb = np.where(wsum > 0, vnum / np.maximum(wsum, 1e-12), 0).astype(np.complex64)

        band = d[:, chan_lo:chan_lo + full_n_chan, :]
        for p in range(npol):
            band[:, :, p] = np.where(hany, vb, band[:, :, p])
        ms.putcol(args.out_col, d, startrow=sr, nrow=nt, rowincr=n_baseline)

        if args.weight_frac is not None:
            # down-weight inpainted pixels to weight_frac x the row's real WEIGHT (idempotent:
            # scaled off WEIGHT, not the possibly-already-modified WEIGHT_SPECTRUM).
            w_row = ms.getcol('WEIGHT', startrow=sr, nrow=nt, rowincr=n_baseline)   # (nt, npol)
            ws = ms.getcol('WEIGHT_SPECTRUM', startrow=sr, nrow=nt, rowincr=n_baseline)
            wsb = ws[:, chan_lo:chan_lo + full_n_chan, :]
            for p in range(npol):
                wsb[:, :, p] = np.where(hany, args.weight_frac * w_row[:, p][:, None], wsb[:, :, p])
            ms.putcol('WEIGHT_SPECTRUM', ws, startrow=sr, nrow=nt, rowincr=n_baseline)

        if args.unflag or args.weight_frac is not None:
            fl = ms.getcol('FLAG', startrow=sr, nrow=nt, rowincr=n_baseline)
            fb = fl[:, chan_lo:chan_lo + full_n_chan, :]
            for p in range(npol):
                fb[:, :, p] = np.where(hany, False, fb[:, :, p])
            ms.putcol('FLAG', fl, startrow=sr, nrow=nt, rowincr=n_baseline)

        written += 1
        if written == 1 or written % 100 == 0:
            log(f"  wrote {written}/{len(groups)} baselines  bl={bl} tlo={tlo} holes={int(hany.sum())}")

    ms.flush(); root.flush(); hf.close()
    log(f"done: {written} baselines -> {args.out_col}{' (FLAG cleared in holes)' if args.unflag else ''}")


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
    ap.add_argument('--weight-frac', type=float, default=None, dest='weight_frac')
    main(ap.parse_args())
