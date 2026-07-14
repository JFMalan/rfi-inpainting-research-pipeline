import argparse
import glob
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth_component(amp, mask, sigma=1.0):
    # same recipe as model/data.py (inlined so this runs torch-free in ASTRO-PY3)
    filled = amp.copy()
    idx = np.arange(amp.shape[1])
    for t in range(amp.shape[0]):
        row = filled[t]
        keep = mask[t] < 0.5
        if keep.sum() < 4:
            filled[t] = row.mean() if keep.any() else 1.0
            continue
        filled[t] = np.interp(idx, idx[keep], row[keep])
    return gaussian_filter(filled, sigma=sigma, mode='nearest').astype(np.float32)


def main(args):
    paths = sorted(glob.glob(args.pattern),
                   key=lambda s: int(re.search(r'_e(\d+)', s).group(1)))
    if not paths:
        raise SystemExit(f"no files match {args.pattern}")
    print(f"{len(paths)} eval snapshots", flush=True)

    eps, m_noisy, m_smooth, mf_smooth = [], [], [], []
    for p in paths:
        d = np.load(p)
        ep = int(re.search(r'_e(\d+)', p).group(1))
        a, b, c = [], [], []
        for i in range(d['obs'].shape[0]):
            amp = d['obs'][i, 0]
            pr = d['pred'][i, 0]
            flags = d['real_flags'][i, 0]
            fake = d['fake_mask'][i, 0] > 0.5
            if fake.sum() < 20:
                continue
            sm = smooth_component(amp, flags, args.sigma)
            trust = (flags < 0.5) & ~fake
            mf = amp[trust].mean() if trust.any() else 1.0
            a.append(np.abs(pr - amp)[fake].mean())
            b.append(np.abs(pr - sm)[fake].mean())
            c.append(np.abs(mf - sm)[fake].mean())
        eps.append(ep)
        m_noisy.append(np.mean(a))
        m_smooth.append(np.mean(b))
        mf_smooth.append(np.mean(c))
        print(f"e{ep:3d}  vs_noisy={m_noisy[-1]:.4f}  vs_smooth={m_smooth[-1]:.4f}  "
              f"meanfill_vs_smooth={mf_smooth[-1]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, m_noisy, 'o-', label='model vs noisy obs (has grain floor)')
    ax.plot(eps, m_smooth, 'o-', label='model vs smooth signal')
    ax.plot(eps, mf_smooth, 's--', label='mean-fill vs smooth signal')
    ax.set_xlabel('epoch')
    ax.set_ylabel('fake-hole amp MAE')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"-> {out}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sigma', type=float, default=1.0)
    main(ap.parse_args())
