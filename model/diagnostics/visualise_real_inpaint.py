import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def green(mask2d):
    rgba = np.zeros((*mask2d.T.shape, 4), np.float32)
    rgba[mask2d.T > 0] = [0.0, 1.0, 0.2, 0.85]
    return rgba


def main(args):
    d = np.load(args.input)
    data, phase, flags, pred = d['data'], d['phase'], d['flags'], d['pred']
    band_min = float(d['band_min']); band_max = float(d['band_max'])
    idxs = d['idxs'] if 'idxs' in d else np.arange(len(data))
    n = min(args.n_show, len(data))
    n_t = data.shape[1]

    titles = ["observed amp (raw, RFI)", "inpainted amp (RFI filled)", "RFI mask",
              "observed phase", "inpainted phase"]
    fig, axes = plt.subplots(n, 5, figsize=(4 * 5, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        dt, ph, fl, pr = data[r], phase[r], flags[r], pred[r]
        unflag = fl < 0.5
        vmin = np.percentile(dt[unflag], 1) if unflag.any() else float(dt.min())
        vmax = np.percentile(dt[unflag], 99) if unflag.any() else float(dt.max())
        filled = np.where(fl > 0.5, pr[0], dt)
        pred_ph = np.arctan2(pr[2], pr[1])
        filled_ph = np.where(fl > 0.5, pred_ph, ph)
        ext = [0, n_t, band_min, band_max]

        axes[r, 0].imshow(dt.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 1].imshow(filled.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 2].imshow(dt.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 2].imshow(green(fl), aspect='auto', origin='lower', extent=ext)
        axes[r, 3].imshow(ph.T, aspect='auto', origin='lower', extent=ext, vmin=-np.pi, vmax=np.pi, cmap='twilight')
        axes[r, 4].imshow(filled_ph.T, aspect='auto', origin='lower', extent=ext, vmin=-np.pi, vmax=np.pi, cmap='twilight')
        axes[r, 0].set_ylabel(f"baseline {int(idxs[r])}\nFreq (MHz)\nflag={fl.mean():.2f}\n"
                              f"scale[{vmin:.2f},{vmax:.2f}]", fontsize=7)
        if r == 0:
            for j, t in enumerate(titles):
                axes[r, j].set_title(t, fontsize=9)
        for ax in axes[r]:
            ax.tick_params(labelsize=6); ax.set_xlabel("Time bin", fontsize=7)

    plt.tight_layout()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"-> {out}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--n-show', type=int, default=6, dest='n_show')
    main(ap.parse_args())
