import argparse
import numpy as np
from pathlib import Path
from casacore.tables import table


def main(args):
    ms = table(args.ms, readonly=True)
    cols = ms.colnames()
    col = 'CORRECTED_DATA' if 'CORRECTED_DATA' in cols else 'DATA'
    data = ms.getcol(col)    # (n_row, n_chan, n_pol)
    flags = ms.getcol('FLAG')
    times = ms.getcol('TIME')
    ms.close()

    freqs_table = table(args.ms + '/SPECTRAL_WINDOW')
    freqs = freqs_table.getcol('CHAN_FREQ')[0]
    freqs_table.close()

    amp = np.abs(data).mean(axis=2).astype(np.float32)  # (n_row, n_chan)
    flagged = flags.any(axis=2)                          # (n_row, n_chan)

    unique_times = np.unique(times)
    n_time = len(unique_times)
    n_chan = amp.shape[1]
    n_baseline = amp.shape[0] // n_time

    amp_3d = np.ma.array(
        amp.reshape(n_time, n_baseline, n_chan),
        mask=flagged.reshape(n_time, n_baseline, n_chan)
    )
    # baseline average; fully-flagged time-channel cells become 0
    waterfall = amp_3d.mean(axis=1).filled(0.0).astype(np.float32)

    # track which cells had no valid baselines (all flagged)
    all_flagged = flagged.reshape(n_time, n_baseline, n_chan).all(axis=1)
    flag_waterfall = all_flagged.astype(np.float32)

    print(f"using column  : {col}")
    print(f"waterfall shape: {waterfall.shape}  freq: {freqs[0]/1e6:.1f}-{freqs[-1]/1e6:.1f} MHz")
    print(f"flagged cells  : {flag_waterfall.mean()*100:.1f}%")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out) + '.npy', waterfall)
    np.save(str(out) + '.flags.npy', flag_waterfall)
    np.save(str(out) + '.meta.npy', np.array([freqs[0] / 1e6, freqs[-1] / 1e6]))
    print(f"saved -> {out}.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
