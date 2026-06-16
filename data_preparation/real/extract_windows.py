import argparse

import numpy as np
import h5py
from casacore.tables import table
from rfi_bands import LBAND_PERSISTENT_MHZ
from extract_ms import divisive_norm, extract_patches


def main(args):
    base = table(args.ms, readonly=True, ack=False)
    if args.field is not None:
        base = base.query(f"FIELD_ID == {args.field}")
    all_times = np.unique(base.getcol('TIME'))
    ntime = len(all_times)

    cols = base.colnames()
    col = args.column if args.column else ('CORRECTED_DATA' if 'CORRECTED_DATA' in cols else 'DATA')
    print(f"column {col}  total timestamps {ntime}", flush=True)

    ft = table(args.ms + '/SPECTRAL_WINDOW', ack=False)
    freqs = ft.getcol('CHAN_FREQ')[0] / 1e6
    ft.close()
    chan_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    freqs = freqs[chan_mask]
    n_chan = int(chan_mask.sum())
    persistent = np.zeros(n_chan, bool)
    for flo, fhi in LBAND_PERSISTENT_MHZ:
        persistent |= (freqs >= flo) & (freqs <= fhi)

    pt, pf = args.patch_time, args.patch_freq
    st, sf = args.stride_time, args.stride_freq
    starts = np.linspace(0, max(ntime - pt, 0), args.n_windows).astype(int)
    print(f"windows at timestamps: {list(starts)}", flush=True)

    A_data, A_raw, A_flag, A_fmin, A_fmax, A_win = [], [], [], [], [], []

    for wi, ts in enumerate(starts):
        hi = min(ts + pt, ntime)
        win = base.query(f"TIME >= {all_times[ts]} AND TIME <= {all_times[hi-1]}")
        data = win.getcol(col)[:, chan_mask, :]
        flags = win.getcol('FLAG')[:, chan_mask, :]
        times = win.getcol('TIME')
        a1 = win.getcol('ANTENNA1'); a2 = win.getcol('ANTENNA2')
        win.close()

        amp = np.abs(data).mean(axis=2).astype(np.float32)
        flagged = flags.any(axis=2)
        flagged[:, persistent] = True

        ut = np.unique(times); nt = len(ut); nbl = amp.shape[0] // nt
        amp = amp[:nt * nbl].reshape(nt, nbl, n_chan)
        flagged = flagged[:nt * nbl].reshape(nt, nbl, n_chan)
        a1 = a1[:nbl]; a2 = a2[:nbl]
        del data, flags

        wcount = 0
        for bl in range(nbl):
            if a1[bl] == a2[bl]:
                continue
            wf = amp[:, bl, :]; fm = flagged[:, bl, :]
            if fm.mean() > args.max_bl_flag_frac:
                continue
            wfn = divisive_norm(wf, fm, smooth_bins=args.smooth_bins)
            pats, fpats, offs = extract_patches(wfn, fm, pt, pf, st, sf,
                                                args.max_flag_frac, args.max_patches_per_bl)
            if not pats:
                continue
            raws, _, _ = extract_patches(wf, fm, pt, pf, st, sf,
                                         args.max_flag_frac, args.max_patches_per_bl)
            A_data.extend(pats); A_raw.extend(raws); A_flag.extend(fpats)
            for o in offs:
                A_fmin.append(freqs[o]); A_fmax.append(freqs[min(o + pf - 1, len(freqs) - 1)])
            A_win.extend([wi] * len(pats))
            wcount += len(pats)
        print(f"  window {wi} (ts {ts}): {nt} timestamps, {wcount} patches", flush=True)

    base.close()
    if not A_data:
        raise SystemExit("no patches extracted")

    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('data', data=np.stack(A_data).astype(np.float32))
        hf.create_dataset('data_raw', data=np.stack(A_raw).astype(np.float32))
        hf.create_dataset('flags', data=np.stack(A_flag).astype(np.float32))
        hf.create_dataset('freq_min_patch', data=np.array(A_fmin, np.float32))
        hf.create_dataset('freq_max_patch', data=np.array(A_fmax, np.float32))
        hf.create_dataset('window_id', data=np.array(A_win, np.int32))
        hf.attrs['n_patches'] = len(A_data)
        hf.attrs['n_windows'] = args.n_windows
    print(f"total {len(A_data)} patches across {args.n_windows} windows -> {args.output}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--column', default='DATA')
    ap.add_argument('--field', type=int, default=None)
    ap.add_argument('--n-windows', type=int, default=5)
    ap.add_argument('--freq-min', type=float, default=900.0)
    ap.add_argument('--freq-max', type=float, default=1650.0)
    ap.add_argument('--patch-time', type=int, default=256)
    ap.add_argument('--patch-freq', type=int, default=256)
    ap.add_argument('--stride-time', type=int, default=64)
    ap.add_argument('--stride-freq', type=int, default=64)
    ap.add_argument('--max-patches-per-bl', type=int, default=8)
    ap.add_argument('--max-flag-frac', type=float, default=1.0)
    ap.add_argument('--max-bl-flag-frac', type=float, default=1.0)
    ap.add_argument('--smooth-bins', type=int, default=64)
    main(ap.parse_args())
