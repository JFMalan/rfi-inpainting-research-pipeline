import argparse

import h5py
import numpy as np
from scipy.ndimage import uniform_filter


def tile_grain(amp, flags):
    # 5x5 high-pass residual std on pixels whose whole 5x5 window is unflagged,
    # same statistic as metrics.noise_floor_ratio's known-region term
    hp = amp - uniform_filter(amp, 5, mode='nearest')
    clear = uniform_filter(flags, 5, mode='nearest') < 1e-6
    if clear.sum() < 100:
        clear = flags < 0.5          # fall back to flag-edge-contaminated pixels
    if clear.sum() < 100:
        return None
    return float(hp[clear].std())


def summarise(name, path, amp_key, flag_key, cap):
    hf = h5py.File(path, 'r')
    n = hf[amp_key].shape[0]
    idx = np.linspace(0, n - 1, min(cap, n)).astype(int)
    grains, levels = [], []
    for u in idx:
        amp = hf[amp_key][u].astype(np.float32)
        flags = hf[flag_key][u].astype(np.float32)
        g = tile_grain(amp, flags)
        if g is None:
            continue
        grains.append(g)
        levels.append(float(np.median(amp[flags < 0.5])))
    hf.close()
    grains, levels = np.array(grains), np.array(levels)
    q = np.percentile(grains, [10, 50, 90])
    print(f"{name}: {len(grains)} tiles  grain p10/p50/p90 = "
          f"{q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}   median amp level {np.median(levels):.3f}",
          flush=True)
    return q[1]


def main(args):
    g_real = summarise('real', args.real, 'data', 'flags', args.cap)
    g_sim = summarise('sim ', args.sim, 'clean', 'mask', args.cap)
    print(f"\nnormalized-grain mismatch: real/sim = {g_real / g_sim:.1f}x")
    print("interpretation: normalized grain ~ sigma/<amp>; while signal-dominated it scales")
    print("inversely with sky flux, saturating near ~0.5 once noise-dominated. a first-cut")
    print(f"flux divisor is the mismatch factor ({g_real / g_sim:.0f}); verify by re-running")
    print("this on one matched-config sim run before committing the full set.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--sim', required=True)
    ap.add_argument('--cap', type=int, default=200)
    main(ap.parse_args())
