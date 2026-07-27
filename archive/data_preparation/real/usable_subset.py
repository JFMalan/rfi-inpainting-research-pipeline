import argparse

import numpy as np
from casacore.tables import table


def main(args):
    ms = table(args.ms, readonly=True, ack=False)
    if args.field is not None:
        ms = ms.query(f"FIELD_ID == {args.field}")

    ft = table(args.ms + '/SPECTRAL_WINDOW', ack=False)
    freqs = ft.getcol('CHAN_FREQ')[0] / 1e6
    ft.close()
    cmask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    freqs = freqs[cmask]
    nchan = int(cmask.sum())

    times_all = ms.getcol('TIME')
    ut = np.unique(times_all)
    ntime = min(len(ut), args.max_time)
    a1 = ms.getcol('ANTENNA1'); a2 = ms.getcol('ANTENNA2')
    nbl = ms.nrows() // len(ut)
    a1 = a1[:nbl]; a2 = a2[:nbl]
    print(f"timestamps {ntime}  baselines {nbl}  channels {nchan}", flush=True)

    flag_tf = np.zeros((ntime, nchan))         # per (time,chan) over baselines, for bad-time id
    bl_flag = np.zeros(nbl)                     # per-baseline mean flag
    bl_cnt = np.zeros(nbl)
    block = nbl * 50
    n_keep = ntime * nbl
    for r0 in range(0, n_keep, block):
        nr = min(block, n_keep - r0)
        fl = ms.getcol('FLAG', r0, nr)[:, cmask, :].any(axis=2)   # (nr, nchan)
        for k in range(nr):
            bl = (r0 + k) % nbl
            tt = (r0 + k) // nbl
            bl_flag[bl] += fl[k].mean(); bl_cnt[bl] += 1
            flag_tf[tt] += fl[k]
        print(f"  read {r0}/{n_keep}", flush=True)
    ms.close()
    bl_flag = bl_flag / np.maximum(bl_cnt, 1)
    flag_tf /= nbl

    auto = a1 == a2
    dead = bl_flag > args.dead_thresh
    print(f"\nautocorr baselines: {auto.sum()}")
    print(f"dead baselines (>{args.dead_thresh:.0%} flagged): {dead.sum()}")

    per_time = flag_tf.mean(axis=1)
    bad_time = per_time > args.bad_time_thresh
    print(f"bad timestamps (>{args.bad_time_thresh:.0%} flagged across band): {bad_time.sum()}/{ntime}")

    usable_bl = ~auto & ~dead
    bf = bl_flag[usable_bl]
    print(f"\n--- flag fraction over {usable_bl.sum()} usable (non-auto, non-dead) baselines ---")
    for p in [5, 10, 25, 50, 75, 90]:
        print(f"  p{p:2d}: {np.percentile(bf, p):.3f}")
    print(f"  mean {bf.mean():.3f}")
    for th in [0.2, 0.3, 0.4, 0.5, 0.6]:
        print(f"  baselines <{th:.0%} flagged: {(bf < th).sum()}  ({100*(bf<th).mean():.1f}% of usable)")

    print("\nper-channel flag frac over usable baselines (every 32nd ch):")
    # recompute per-channel on usable baselines would need re-read; report band hint instead
    print("  (see flag_diagnostics.png for per-channel profile)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--field', type=int, default=None)
    ap.add_argument('--max-time', type=int, default=512)
    ap.add_argument('--freq-min', type=float, default=900.0)
    ap.add_argument('--freq-max', type=float, default=1650.0)
    ap.add_argument('--dead-thresh', type=float, default=0.95)
    ap.add_argument('--bad-time-thresh', type=float, default=0.95)
    main(ap.parse_args())
