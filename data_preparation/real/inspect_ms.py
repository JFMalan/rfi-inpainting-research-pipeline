import argparse

import numpy as np
from casacore.tables import table

PERSISTENT = [(930, 960, "GSM-900"), (1170, 1300, "GPS/GLONASS"), (1525, 1630, "sat downlink")]


def main(args):
    ms = args.ms
    print(f"opening {ms}", flush=True)
    t = table(ms, readonly=True, ack=False)
    a1 = t.getcol('ANTENNA1'); a2 = t.getcol('ANTENNA2')
    cross = a1 != a2
    print(f"rows {t.nrows()}  cross {cross.sum()}  auto {(~cross).sum()}", flush=True)

    sw = table(ms + "/SPECTRAL_WINDOW", ack=False)
    cf = sw.getcol('CHAN_FREQ')[0] / 1e6
    sw.close()
    nchan = len(cf)

    idx = np.where(cross)[0]
    sample = idx[np.linspace(0, len(idx) - 1, args.nsample).astype(int)]
    print(f"sampling {len(sample)} cross-corr rows spread across the MS", flush=True)

    ff = np.empty(len(sample))
    chan_flag_sum = np.zeros(nchan)
    chan_flag_cnt = 0
    unfl_vals, flg_vals = [], []
    for j, i in enumerate(sample):
        fcell = t.getcell('FLAG', int(i))          # (nchan, npol)
        ff[j] = fcell.mean()
        chf_row = fcell.mean(axis=1)
        chan_flag_sum += chf_row
        chan_flag_cnt += 1
        dcell = np.abs(t.getcell(args.column, int(i))).mean(axis=1)   # (nchan,) pol-mean amp
        rowflag = fcell.any(axis=1)
        unfl_vals.append(dcell[~rowflag])
        flg_vals.append(dcell[rowflag])
        if (j + 1) % 500 == 0:
            print(f"  {j + 1}/{len(sample)}", flush=True)

    print(f"\ncross-corr FLAG frac: mean {ff.mean():.3f} median {np.median(ff):.3f} "
          f"min {ff.min():.3f} max {ff.max():.3f}")
    print(f"  >90% flagged: {(ff > 0.9).mean():.3f}   <50% (usable): {(ff < 0.5).mean():.3f}")

    unfl = np.concatenate(unfl_vals) if unfl_vals else np.array([])
    flg = np.concatenate(flg_vals) if flg_vals else np.array([])
    if unfl.size:
        print(f"\n{args.column} amp unflagged: mean {unfl.mean():.2f} p5 {np.percentile(unfl, 5):.2f} "
              f"p50 {np.percentile(unfl, 50):.2f} p95 {np.percentile(unfl, 95):.2f}")
    if flg.size:
        print(f"{args.column} amp flagged  : mean {flg.mean():.2f} p50 {np.percentile(flg, 50):.2f} "
              f"p95 {np.percentile(flg, 95):.2f}  (>> unflagged if flags catch RFI)")

    chf = chan_flag_sum / max(chan_flag_cnt, 1)
    print("\nper-channel flag frac vs known RFI bands (every 32nd ch):")
    for c in range(0, nchan, 32):
        tag = next((nm for lo, hi, nm in PERSISTENT if lo <= cf[c] <= hi), "")
        bar = '#' * int(chf[c] * 40)
        print(f"  {cf[c]:7.1f} {chf[c]:.2f} {bar:<40} {tag}")

    inband = np.zeros(nchan, bool)
    for lo, hi, _ in PERSISTENT:
        inband |= (cf >= lo) & (cf <= hi)
    print(f"\nflag frac INSIDE known RFI bands : {chf[inband].mean():.3f}")
    print(f"flag frac OUTSIDE known RFI bands: {chf[~inband].mean():.3f}")
    print(f"clean channels (<20% flagged): {(chf < 0.2).sum()}/{nchan}")
    t.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--column', default='DATA')
    ap.add_argument('--nsample', type=int, default=3000)
    main(ap.parse_args())
