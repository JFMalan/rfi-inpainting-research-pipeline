"""
Step 1 of extraction: reads MS with casacore, saves waterfall + metadata as numpy.
Run in stimela_meqtrees container (has casacore, no h5py).
Output: <output>.npy (waterfall) and <output>.meta.npy (freq_min, freq_max)
"""
import argparse
import numpy as np
from pathlib import Path
from casacore.tables import table


def main(args):
    ms = table(args.ms, readonly=True)
    data = ms.getcol('DATA')    # (n_row, n_chan, n_pol)
    flags = ms.getcol('FLAG')
    times = ms.getcol('TIME')
    ms.close()

    freqs_table = table(args.ms + '/SPECTRAL_WINDOW')
    freqs = freqs_table.getcol('CHAN_FREQ')[0]
    freqs_table.close()

    amp = np.abs(data).mean(axis=2).astype(np.float32)
    flagged = flags.any(axis=2)

    unique_times = np.unique(times)
    n_time = len(unique_times)
    n_chan = amp.shape[1]
    n_baseline = amp.shape[0] // n_time

    amp_3d = np.ma.array(
        amp.reshape(n_time, n_baseline, n_chan),
        mask=flagged.reshape(n_time, n_baseline, n_chan)
    )
    waterfall = amp_3d.mean(axis=1).filled(0.0).astype(np.float32)
    print(f"waterfall shape: {waterfall.shape}  freq: {freqs[0]/1e6:.1f}-{freqs[-1]/1e6:.1f} MHz")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out) + '.npy', waterfall)
    np.save(str(out) + '.meta.npy', np.array([freqs[0] / 1e6, freqs[-1] / 1e6]))
    print(f"saved -> {out}.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', required=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
