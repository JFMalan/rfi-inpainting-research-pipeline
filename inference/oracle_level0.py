import argparse
import time

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
    chan_hi = chan_lo + int(hf.attrs['full_n_chan'])
    has_tlo = 'time_lo' in hf
    f_lo_h5 = float(hf.attrs['freq_min_mhz'])
    f_hi_h5 = float(hf.attrs['freq_max_mhz'])

    root = table(args.ms, readonly=True, ack=False)
    times = root.getcol('TIME')
    ant1 = root.getcol('ANTENNA1')
    ant2 = root.getcol('ANTENNA2')
    n_row = root.nrows()
    uniq_t = np.unique(times)
    n_time = len(uniq_t)
    if n_row % n_time != 0:
        raise RuntimeError(f"n_row {n_row} not divisible by n_time {n_time} - row map unsafe")
    n_baseline = n_row // n_time
    tb = times[:n_time * n_baseline].reshape(n_time, n_baseline)
    if not np.all(tb.max(axis=1) == tb.min(axis=1)):
        raise RuntimeError("MS rows not time-major - row map unsafe")
    log(f"MS {n_row} rows  n_time={n_time}  n_baseline={n_baseline}")

    sw = table(args.ms + '/SPECTRAL_WINDOW', ack=False)
    freqs = sw.getcol('CHAN_FREQ')[0] / 1e6
    sw.close()
    f_lo_ms = float(freqs[chan_lo])
    f_hi_ms = float(freqs[chan_hi - 1])
    df = abs(freqs[1] - freqs[0])
    chan_ok = abs(f_lo_ms - f_lo_h5) < df and abs(f_hi_ms - f_hi_h5) < df
    log(f"channel map chan_lo={chan_lo} chan_hi={chan_hi}  "
        f"h5 {f_lo_h5:.3f}-{f_hi_h5:.3f} MHz  ms {f_lo_ms:.3f}-{f_hi_ms:.3f} MHz  ok={chan_ok}")

    n_units = hf[hole_key].shape[0]
    cap = n_units if args.max_units is None else min(args.max_units, n_units)
    log(f"verifying row map for {cap}/{n_units} units")

    bad_ant = bad_time = bad_const = 0
    for u in range(cap):
        bl = int(hf['baseline_id'][u]); nt = int(hf['native_n_time'][u])
        tlo = int(hf['time_lo'][u]) if has_tlo else 0
        sr = tlo * n_baseline + bl
        rows = sr + np.arange(nt) * n_baseline
        a1 = ant1[rows]; a2 = ant2[rows]; tr = times[rows]
        if not (np.all(a1 == a1[0]) and np.all(a2 == a2[0])):
            bad_const += 1
        if 'ant1' in hf and 'ant2' in hf:
            if int(a1[0]) != int(hf['ant1'][u]) or int(a2[0]) != int(hf['ant2'][u]):
                bad_ant += 1
        if not np.all(np.diff(tr) > 0):
            bad_time += 1
        if u == 0 or (u + 1) % 200 == 0:
            log(f"  checked {u + 1}/{cap}  bad_const={bad_const} bad_ant={bad_ant} bad_time={bad_time}")

    root.close()
    log(f"ROW MAP: units={cap}  not-constant-baseline={bad_const}  "
        f"ant-mismatch={bad_ant}  time-not-monotonic={bad_time}  channel_ok={chan_ok}")
    map_ok = (bad_const == 0 and bad_ant == 0 and bad_time == 0 and chan_ok)
    log(f"ROW MAP {'VERIFIED' if map_ok else 'FAILED - mechanics bug'}")

    if args.verify_only:
        hf.close()
        return

    root = table(args.ms, readonly=False, ack=False)
    ensure_column(root, args.out_col, 'DATA')
    log(f"identity write of native DATA into holes -> {args.out_col}")

    max_diff = 0.0
    written = 0
    for u in range(cap):
        bl = int(hf['baseline_id'][u]); nt = int(hf['native_n_time'][u])
        tlo = int(hf['time_lo'][u]) if has_tlo else 0
        hole = resize(hf[hole_key][u].astype(np.float32), (nt, chan_hi - chan_lo),
                      order=0, mode='edge', preserve_range=True) > 0.5
        sr = tlo * n_baseline + bl

        src = root.getcol('DATA', startrow=sr, nrow=nt, rowincr=n_baseline)
        d = root.getcol(args.out_col, startrow=sr, nrow=nt, rowincr=n_baseline)
        sband = src[:, chan_lo:chan_hi, :]
        band = d[:, chan_lo:chan_hi, :]
        for p in range(d.shape[2]):
            band[:, :, p] = np.where(hole, sband[:, :, p], band[:, :, p])
        root.putcol(args.out_col, d, startrow=sr, nrow=nt, rowincr=n_baseline)

        if args.unflag:
            fl = root.getcol('FLAG', startrow=sr, nrow=nt, rowincr=n_baseline)
            fb = fl[:, chan_lo:chan_hi, :]
            for p in range(fl.shape[2]):
                fb[:, :, p] = np.where(hole, False, fb[:, :, p])
            root.putcol('FLAG', fl, startrow=sr, nrow=nt, rowincr=n_baseline)

        if u < 20:
            chk = root.getcol(args.out_col, startrow=sr, nrow=nt, rowincr=n_baseline)
            max_diff = max(max_diff, float(np.abs(chk - src).max()))

        written += 1
        if written == 1 or written % 100 == 0:
            log(f"  wrote {written}/{cap}  bl={bl} tlo={tlo} holes={int(hole.sum())}")

    root.flush(); root.close(); hf.close()
    log(f"done: {written} units -> {args.out_col}  readback max|ORACLE0-DATA|={max_diff:.3e} (expect 0)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--out-col', default='ORACLE0_DATA', dest='out_col')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--unflag', action='store_true')
    ap.add_argument('--verify-only', action='store_true', dest='verify_only')
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    main(ap.parse_args())
