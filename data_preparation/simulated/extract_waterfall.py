import argparse
import numpy as np
from pathlib import Path
from casacore.tables import table


def main(args):
    ms = table(args.ms, readonly=True)
    cols = ms.colnames()
    col = 'CORRECTED_DATA' if 'CORRECTED_DATA' in cols else 'DATA'
    try:
        ms.getcell(col, 0)
    except Exception:
        col = 'DATA'

    data  = ms.getcol(col)
    flags = ms.getcol('FLAG')
    times = ms.getcol('TIME')
    ant1  = ms.getcol('ANTENNA1')
    ant2  = ms.getcol('ANTENNA2')
    ms.close()

    freqs_table = table(args.ms + '/SPECTRAL_WINDOW')
    freqs = freqs_table.getcol('CHAN_FREQ')[0] / 1e6
    freqs_table.close()

    chan_mask = (freqs >= args.freq_min) & (freqs <= args.freq_max)
    data  = data[:, chan_mask, :]
    flags = flags[:, chan_mask, :]
    freqs = freqs[chan_mask]

    amp     = np.abs(data).mean(axis=2).astype(np.float32)
    flagged = flags.any(axis=2)

    unique_times = np.unique(times)
    n_time       = len(unique_times)
    n_baseline   = amp.shape[0] // n_time
    autocorr     = (ant1 == ant2)[:n_baseline]

    amp     = amp[:n_time * n_baseline].reshape(n_time, n_baseline, amp.shape[1])
    flagged = flagged[:n_time * n_baseline].reshape(n_time, n_baseline, flagged.shape[1])

    # baseline-average (exclude autocorrelations)
    bl_mask = ~autocorr
    waterfall = np.ma.array(amp[:, bl_mask, :], mask=flagged[:, bl_mask, :]).mean(axis=1).filled(0.0)
    flag_map  = flagged[:, bl_mask, :].all(axis=1)

    print(f"column    : {col}")
    print(f"freq range: {freqs[0]:.1f}–{freqs[-1]:.1f} MHz  ({len(freqs)} channels)")
    print(f"time bins : {n_time}")
    print(f"baselines : {bl_mask.sum()} (excl. {autocorr.sum()} autocorr)")
    print(f"waterfall : {waterfall.shape}  flag frac: {flag_map.mean():.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out) + '.npy', waterfall.astype(np.float32))
    np.save(str(out) + '.meta.npy', np.array([freqs[0], freqs[-1]]))
    np.save(str(out) + '.flags.npy', flag_map)
    print(f"saved -> {out}.npy  {out}.meta.npy  {out}.flags.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms',       required=True)
    parser.add_argument('--output',   required=True)
    parser.add_argument('--freq-min', type=float, default=900.0)
    parser.add_argument('--freq-max', type=float, default=1650.0)
    main(parser.parse_args())
