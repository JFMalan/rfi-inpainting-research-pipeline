import argparse
import time
from pathlib import Path

import h5py
import numpy as np

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def main(args):
    noisy = h5py.File(args.noisy, 'r')
    clean = h5py.File(args.clean, 'r')
    n = noisy['clean'].shape[0]

    cbl, ctl, cfl = clean['baseline_id'][:], clean['time_lo'][:], clean['freq_lo'][:]
    cmap = {(int(cbl[r]), int(ctl[r]), int(cfl[r])): r for r in range(clean['clean'].shape[0])}
    log(f"noisy rows={n}  clean rows={len(cmap)}  building paired target (noise-free amp / input divisor)")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    chunk = 256
    with h5py.File(out, 'w') as g:
        for k in noisy.keys():
            g.create_dataset(k, shape=noisy[k].shape, dtype=noisy[k].dtype)
        for k, v in noisy.attrs.items():
            g.attrs[k] = v

        for k in noisy.keys():
            if k == 'clean':
                continue
            src = noisy[k]
            for s in range(0, src.shape[0], chunk):
                g[k][s:s + chunk] = src[s:s + chunk]
        log("copied input fields (corrupted/mask/phase/divisor/positions) from noisy")

        bl, tl, fl = noisy['baseline_id'][:], noisy['time_lo'][:], noisy['freq_lo'][:]
        miss = 0
        for r in range(n):
            key = (int(bl[r]), int(tl[r]), int(fl[r]))
            cr = cmap.get(key)
            if cr is None:
                g['clean'][r] = noisy['clean'][r]; miss += 1; continue
            div1 = noisy['dn_divisor'][r]
            div0 = clean['dn_divisor'][cr]
            g['clean'][r] = (clean['clean'][cr] * div0 / np.maximum(div1, 1e-6)).astype(np.float32)
            if (r + 1) % 500 == 0:
                log(f"  {r + 1}/{n}  unmatched={miss}")
        log(f"done  unmatched={miss}/{n}  -> {out}")
    noisy.close(); clean.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--noisy', required=True, help='dataset.h5 at target noise (input/context source)')
    ap.add_argument('--clean', required=True, help='dataset.h5 at NOISE_SCALE=0 (noise-free amp target source)')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
