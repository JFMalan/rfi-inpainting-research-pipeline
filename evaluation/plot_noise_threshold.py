import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEFD_MHZ = np.array([856, 900, 950, 1000, 1100, 1280, 1400, 1450, 1550, 1600, 1650, 1712])
SEFD_JY = np.array([560, 510, 450, 420, 390, 390, 400, 420, 450, 470, 500, 560])
DNU = 856e6 / 1024
DT = 8.0


def sigma_of(scale, fmin=900, fmax=1650):
    f = np.linspace(fmin, fmax, 256)
    sefd = np.interp(f, SEFD_MHZ, SEFD_JY).mean()
    return scale * sefd / np.sqrt(2.0 * DNU * DT)


def best_metrics(run_dir):
    log = json.loads((run_dir / 'log.json').read_text())
    ev = [e for e in log if 'amp_mae' in e]
    if not ev:
        return None
    b = min(ev, key=lambda e: e['amp_mae'])
    mf = float(np.median([e['amp_mf'] for e in ev]))
    return b['amp_mae'], mf, b.get('psnr', np.nan), b.get('epoch', -1)


def median_vis_amp(clean_h5):
    with h5py.File(clean_h5, 'r') as f:
        for k in ('data', 'amp', 'vis', 'waterfall', 'visibilities'):
            if k in f:
                a = f[k][: min(200, f[k].shape[0])]
                a = np.abs(a) if np.iscomplexobj(a) else np.asarray(a)
                return float(np.median(a[a > 0])) if (a > 0).any() else float(np.median(a))
    return None


def main(args):
    scales = [float(s) for s in args.scales.split()]
    root = Path(args.runs_root)
    rows = []
    for s in scales:
        tag = f"thr_n{int(round(s * 1000)):04d}"
        rd = root / f"phase1_{tag}"
        m = best_metrics(rd) if (rd / 'log.json').exists() else None
        if m is None:
            print(f"skip scale={s}: no log.json in {rd}", flush=True)
            continue
        rows.append((s, sigma_of(s), *m))
        print(f"scale={s} sigma={sigma_of(s):.4f}Jy  model_mae={m[0]:.5f}  mf_mae={m[1]:.5f}  "
              f"psnr={m[2]:.2f}  epoch={m[3]}", flush=True)
    if not rows:
        raise SystemExit("no runs found")
    rows.sort()
    sc = np.array([r[0] for r in rows])
    model = np.array([r[2] for r in rows])
    mf = np.array([r[3] for r in rows])
    psnr = np.array([r[4] for r in rows])

    vamp = median_vis_amp(args.clean_h5) if args.clean_h5 and Path(args.clean_h5).exists() else None
    snr = (vamp / np.array([r[1] for r in rows])) if vamp else None
    if vamp:
        print(f"median |V|={vamp:.4f} Jy (from {args.clean_h5}); SNR per level: "
              + ", ".join(f"{s}:{sn:.1f}" for s, sn in zip(sc, snr)), flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(sc, model, 'o-', label='model (in-hole MAE)', color='C0')
    ax[0].plot(sc, mf, 's--', label='mean-fill baseline', color='C3')
    ax[0].fill_between(sc, model, mf, where=(model < mf), alpha=0.15, color='C0')
    ax[0].set_xscale('log'); ax[0].set_xlabel('thermal noise (x MeerKAT SEFD)')
    ax[0].set_ylabel('amplitude MAE in holes'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title('trainability: model vs mean-fill')

    cross = None
    adv = mf - model
    for i in range(len(sc) - 1):
        if adv[i] > 0 >= adv[i + 1]:
            t = adv[i] / (adv[i] - adv[i + 1])
            cross = sc[i] * (sc[i + 1] / sc[i]) ** t
            break
    if cross:
        for a in ax:
            a.axvline(cross, color='k', ls=':', lw=1)
        ax[0].annotate(f'threshold ~{cross:.2f}x SEFD', (cross, ax[0].get_ylim()[1]),
                       fontsize=8, ha='center', va='top')

    ax[1].plot(sc, psnr, 'o-', color='C2')
    ax[1].set_xscale('log'); ax[1].set_xlabel('thermal noise (x MeerKAT SEFD)')
    ax[1].set_ylabel('PSNR in holes (dB)'); ax[1].grid(alpha=0.3); ax[1].set_title('PSNR vs noise')

    if snr is not None:
        secax = ax[0].secondary_xaxis('top')
        secax.set_xscale('log')
        secax.set_xticks(sc)
        secax.set_xticklabels([f"{v:.0f}" for v in snr], fontsize=7)
        secax.set_xlabel('per-baseline SNR', fontsize=8)

    plt.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"-> {out}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', required=True)
    ap.add_argument('--scales', required=True)
    ap.add_argument('--clean-h5', default='')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
