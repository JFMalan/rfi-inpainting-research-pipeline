import argparse

import h5py
import numpy as np


def tdiff_sigma(V, keep):
    # successive-difference thermal-noise estimator along time (axis 0). adjacent 8s
    # dumps see the same sky, so V[t+1]-V[t] cancels signal and leaves sqrt(2)x the
    # per-component thermal sigma. resize along time would correlate samples and bias low.
    d = V[1:] - V[:-1]
    k = keep[1:] & keep[:-1]
    if k.sum() < 200:
        return None
    dd = d[k]
    var = 0.5 * (dd.real.var() + dd.imag.var())
    return np.sqrt(var / 2.0)


def snr_tdiff(amp, phase, keep):
    V = amp * np.exp(1j * phase)
    sig = np.median(np.abs(V[keep])) if keep.any() else np.nan
    sigma = tdiff_sigma(V, keep)
    return (sig / sigma) if (sigma and sigma > 1e-9) else np.nan


def run_real(path, cap):
    hf = h5py.File(path, 'r')
    n = hf['data'].shape[0]
    nt_native = int(hf['native_n_time'][0]) if 'native_n_time' in hf else None
    idx = np.linspace(0, n - 1, min(cap, n)).astype(int)
    snrs = []
    for u in idx:
        amp = hf['data'][u].astype(np.float64)
        phase = hf['phase'][u].astype(np.float64)
        keep = hf['flags'][u].astype(np.float32) < 0.5
        s = snr_tdiff(amp, phase, keep)
        if np.isfinite(s):
            snrs.append(s)
    hf.close()
    return np.array(snrs), nt_native


def run_sim(path, cap):
    hf = h5py.File(path, 'r')
    n = hf['clean'].shape[0]
    nt_native = int(hf['native_n_time'][0]) if 'native_n_time' in hf else None
    has_tgt = 'amp_target' in hf
    idx = np.linspace(0, n - 1, min(cap, n)).astype(int)
    td, ex = [], []
    for u in idx:
        amp = hf['clean'][u].astype(np.float64)
        phase = hf['phase'][u].astype(np.float64)
        keep = hf['mask'][u].astype(np.float32) < 0.5
        s = snr_tdiff(amp, phase, keep)
        if np.isfinite(s):
            td.append(s)
        if has_tgt:
            tgt = hf['amp_target'][u].astype(np.float64)
            pht = hf['phase_target'][u].astype(np.float64) if 'phase_target' in hf else phase
            dV = amp * np.exp(1j * phase) - tgt * np.exp(1j * pht)
            k = keep
            if k.sum() > 200:
                sigma = np.sqrt(0.5 * (dV[k].real.var() + dV[k].imag.var()))
                sig = np.median(tgt[k])
                if sigma > 1e-9:
                    ex.append(sig / sigma)
    hf.close()
    return np.array(td), np.array(ex), nt_native


def pct(a):
    return '/'.join(f'{v:.1f}' for v in np.percentile(a, [10, 50, 90])) if len(a) else 'n/a'


def main(a):
    r_snr, r_nt = run_real(a.real, a.cap)
    s_td, s_ex, s_nt = run_sim(a.sim, a.cap)
    print(f"real native_n_time={r_nt} (time resized if != {a.tile})  tiles={len(r_snr)}")
    print(f"sim  native_n_time={s_nt} (time resized if != {a.tile})  tiles={len(s_td)}")
    print()
    print(f"{'dataset/method':<28}{'SNR p10/p50/p90':>22}")
    print(f"{'sim  exact (clean-target)':<28}{pct(s_ex):>22}")
    print(f"{'sim  time-diff estimator':<28}{pct(s_td):>22}")
    print(f"{'real time-diff estimator':<28}{pct(r_snr):>22}")
    if len(s_ex) and len(s_td):
        print(f"\nestimator check (sim): time-diff median {np.median(s_td):.1f} vs "
              f"exact {np.median(s_ex):.1f} -> ratio {np.median(s_td)/np.median(s_ex):.2f}")
    if len(s_ex) and len(r_snr):
        print(f"deployment gap: sim exact p50 {np.median(s_ex):.1f} vs real p50 "
              f"{np.median(r_snr):.1f} -> real SNR is {np.median(s_ex)/np.median(r_snr):.1f}x lower")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--sim', required=True)
    ap.add_argument('--cap', type=int, default=200)
    ap.add_argument('--tile', type=int, default=512)
    main(ap.parse_args())
