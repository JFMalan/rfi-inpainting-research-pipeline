import argparse
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from casacore.tables import table
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classical_fill import dpss_basis, dpss_fill

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def blackman_harris(n):
    k = np.arange(n)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    return (a[0] - a[1] * np.cos(2 * np.pi * k / (n - 1))
            + a[2] * np.cos(4 * np.pi * k / (n - 1))
            - a[3] * np.cos(6 * np.pi * k / (n - 1))).astype(np.float32)


def delay_power(V, taper):
    vw = V * taper[None, :]
    vhat = np.fft.fftshift(np.fft.fft(vw, axis=1), axes=1)
    return np.abs(vhat).astype(np.float64) ** 2   # (nt, N)


def main(args):
    if not args.sim:
        # the 'clean' reference here is the MS DATA column, which is RFI-free only on sim
        # (RFI lives in the h5/mask, never in DATA). On real, DATA carries the RFI, so every
        # RFI-removing fill scores hi-ratio ~0 against an RFI-inflated reference — meaningless.
        # Real delay recovery is measured by fakehole_delay_eval.py (fake holes over good data).
        log("SKIP: MS-based delay eval is sim-only (DATA is not a clean reference on real). "
            "Use fakehole_delay_eval.py for the real delay verdict.")
        return
    hole_key = 'mask' if args.sim else 'flags'
    hf = h5py.File(args.h5, 'r')
    chan_lo = int(hf.attrs['chan_lo'])
    N = int(hf.attrs['full_n_chan'])
    fmin = float(hf.attrs['freq_min_mhz']); fmax = float(hf.attrs['freq_max_mhz'])
    has_tlo = 'time_lo' in hf
    has_flo = 'freq_lo' in hf

    cap = hf[hole_key].shape[0] if args.max_units is None else min(args.max_units, hf[hole_key].shape[0])
    bl_arr  = hf['baseline_id'][:cap].astype(int)
    nt_arr  = hf['native_n_time'][:cap].astype(int)
    nc_arr  = hf['native_n_chan'][:cap].astype(int)
    tlo_arr = hf['time_lo'][:cap].astype(int) if has_tlo else np.zeros(cap, int)
    flo_arr = hf['freq_lo'][:cap].astype(int) if has_flo else np.zeros(cap, int)
    groups = defaultdict(list)
    for u in range(cap):
        groups[(bl_arr[u], tlo_arr[u])].append(u)

    root = table(args.ms, readonly=True, ack=False)
    have_inp = args.inp_col in root.colnames()
    times = root.getcol('TIME')
    n_time = len(np.unique(times))
    n_baseline = root.nrows() // n_time
    taper = blackman_harris(N)
    df_hz = (fmax - fmin) * 1e6 / (N - 1)
    tau_us = np.fft.fftshift(np.fft.fftfreq(N, d=df_hz)) * 1e6
    # DPSS classical-baseline configs: a half-width sweep, or a single (hw, lam).
    dpss_cfgs = []
    if args.dpss_sweep:
        for hw in [float(x) for x in args.dpss_sweep.split(',')]:
            dpss_cfgs.append((f'dpss_hw{hw:g}', dpss_basis(N, hw), args.dpss_lam))
    elif args.dpss:
        dpss_cfgs.append(('dpss', dpss_basis(N, args.dpss_hw), args.dpss_lam))
    dpss_labels = [c[0] for c in dpss_cfgs]
    log(f"delay spectrum: {len(groups)} baselines  band {fmin:.1f}-{fmax:.1f} MHz ({N} ch)  "
        f"inpcol={args.inp_col} present={have_inp}  dpss={dpss_labels} (lam={args.dpss_lam})")

    variants = ['clean', 'flagged'] + dpss_labels + (['inpainted'] if have_inp else [])
    P = {k: np.zeros(N, np.float64) for k in variants}
    n_acc = 0
    for gi, ((bl, tlo), units) in enumerate(groups.items()):
        nt = nt_arr[units[0]]
        sr = tlo * n_baseline + bl
        Dc = root.getcol('DATA', startrow=sr, nrow=nt, rowincr=n_baseline)[:, chan_lo:chan_lo + N, :]
        Vc = Dc.mean(axis=2)
        hole = np.zeros((nt, N), bool)
        for u in units:
            nc = nc_arr[u]; flo = flo_arr[u]
            h = resize(hf[hole_key][u].astype(np.float32), (nt, nc), order=0,
                       mode='edge', preserve_range=True) > 0.5
            hole[:, flo:flo + nc] |= h
        Vf = Vc.copy(); Vf[hole] = 0.0
        P['clean']   += delay_power(Vc, taper).mean(axis=0)
        P['flagged'] += delay_power(Vf, taper).mean(axis=0)
        for label, A, lam in dpss_cfgs:
            P[label] += delay_power(dpss_fill(Vc, hole, A, lam), taper).mean(axis=0)
        if have_inp:
            Di = root.getcol(args.inp_col, startrow=sr, nrow=nt, rowincr=n_baseline)[:, chan_lo:chan_lo + N, :]
            P['inpainted'] += delay_power(Di.mean(axis=2), taper).mean(axis=0)
        n_acc += 1
        if gi == 0 or (gi + 1) % 200 == 0:
            log(f"  baseline {gi + 1}/{len(groups)}  holes={int(hole.sum())}")

    root.close(); hf.close()
    for k in P:
        P[k] /= max(n_acc, 1)

    center = N // 2
    hi = np.abs(np.arange(N) - center) > args.fg_bins
    eps = 1e-30
    wclean = P['clean'] + eps   # power weight: emphasise where signal lives, suppress noisy deep-delay bins

    def wlogrmse(kk):
        d = (np.log10(P[kk] + eps) - np.log10(P['clean'] + eps)) ** 2
        return float(np.sqrt((wclean * d).sum() / wclean.sum()))

    def logrmse(kk):
        return float(np.sqrt(np.mean((np.log10(P[kk] + eps) - np.log10(P['clean'] + eps)) ** 2)))

    def hiratio(kk):
        return float(P[kk][hi].sum() / max(P['clean'][hi].sum(), eps))

    log(f"=== delay-space power ({n_acc} baselines; wlogP-RMSE = power-weighted, the robust headline) ===")
    log(f"  {'variant':<12}{'wlogP-RMSE':>12}{'logP-RMSE':>12}{'hi-delay ratio':>16}")
    for kk in [v for v in variants if v != 'clean']:
        log(f"  {kk:<12}{wlogrmse(kk):>12.4f}{logrmse(kk):>12.4f}{hiratio(kk):>16.3f}")

    best_dpss = min(dpss_labels, key=wlogrmse) if dpss_labels else None
    if best_dpss:
        log(f"  best DPSS (lowest wlogP-RMSE): {best_dpss}")
    if have_inp:
        for base in ([best_dpss] if best_dpss else []) + ['flagged']:
            win = wlogrmse('inpainted') < wlogrmse(base) and abs(hiratio('inpainted') - 1) < abs(hiratio(base) - 1)
            log(f"  verdict vs {base}: {'MODEL wins' if win else 'model does NOT clearly beat ' + base} "
                f"(wlogP-RMSE {wlogrmse('inpainted'):.4f} vs {wlogrmse(base):.4f}; "
                f"hi-ratio {hiratio('inpainted'):.3f} vs {hiratio(base):.3f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    for kk in ['clean', 'flagged'] + ([best_dpss] if best_dpss else []) + (['inpainted'] if have_inp else []):
        ax.semilogy(tau_us, P[kk] + eps, label=kk, lw=1.3)
    ax.axvline(tau_us[center + args.fg_bins], color='gray', ls='--', lw=0.8)
    ax.axvline(tau_us[center - args.fg_bins], color='gray', ls='--', lw=0.8)
    ax.set_xlabel('delay (us)'); ax.set_ylabel('power'); ax.legend()
    ax.set_title('delay spectrum (FFT along frequency, per-baseline averaged)')
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches='tight')
    log(f"saved -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--inp-col', default='INPAINTED_DATA', dest='inp_col')
    ap.add_argument('--out', required=True)
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--fg-bins', type=int, default=20, dest='fg_bins')
    ap.add_argument('--dpss', action='store_true', help='add the DPSS classical gap-fill baseline')
    ap.add_argument('--dpss-hw', type=float, default=0.1, dest='dpss_hw',
                    help='DPSS delay half-width as a fraction of the Nyquist delay')
    ap.add_argument('--dpss-lam', type=float, default=0.1, dest='dpss_lam', help='DPSS ridge regularisation')
    ap.add_argument('--dpss-sweep', default=None, dest='dpss_sweep',
                    help='comma list of DPSS half-widths to sweep, e.g. 0.04,0.1,0.2 (lam fixed at --dpss-lam)')
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    main(ap.parse_args())
