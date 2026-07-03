import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import phase2
from data import positional_encoding, build_cond, fake_mask
from diffusion import Diffusion
from unet import UNet
from classical_fill import dpss_basis, dpss_fill

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def blackman_harris(n):
    k = np.arange(n)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    return (a[0] - a[1] * np.cos(2 * np.pi * k / (n - 1))
            + a[2] * np.cos(4 * np.pi * k / (n - 1))
            - a[3] * np.cos(6 * np.pi * k / (n - 1))).astype(np.float64)


def delay_power(V, taper):
    vw = V * taper[None, :]
    return (np.abs(np.fft.fftshift(np.fft.fft(vw, axis=1), axes=1)) ** 2).mean(axis=0)


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    rng = np.random.default_rng(args.seed)

    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    band_min = float(hf.attrs['freq_min_mhz'])
    band_max = float(hf.attrs['freq_max_mhz'])
    n = hf['data'].shape[0]
    split = hf['split'][:] if 'split' in hf else np.ones(n, np.int32)
    has_fpatch = 'freq_min_patch' in hf

    flags_frac = hf['flags'][:].astype(np.float32).mean(axis=(1, 2))
    is_test = split == 1
    fp = flags_frac[is_test]
    log(f"split==1 units={int(is_test.sum())}  their flagfrac "
        f"min/med/max={fp.min():.2f}/{np.median(fp):.2f}/{fp.max():.2f}" if is_test.any() else "no split==1 units")
    test = np.where(is_test & (flags_frac < args.max_flag_frac))[0]
    if args.max_units:
        test = test[:args.max_units]
    log(f"device={dev} ({gpu})  h5={args.h5}  test units={len(test)} (split==1, flagfrac<{args.max_flag_frac})")

    cfg = phase2(predict=args.predict)
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    A = dpss_basis(sz, args.dpss_hw)
    taper = blackman_harris(sz)
    log(f"model loaded; dpss hw={args.dpss_hw} K={A.shape[0]} lam={args.dpss_lam}")

    floors = [None if f in ('none', 'None') else ('auto' if f == 'auto' else float(f))
              for f in args.noise_floors]
    mkeys = [f"model_nf{('none' if f is None else f)}" for f in floors]
    variants = ['truth', 'dpss', 'flagged'] + mkeys
    Pl = {k: [] for k in variants}          # per-tile delay power, for bootstrap CIs
    n_acc = 0
    pe_cache = {}
    bs = args.batch
    with torch.no_grad():
        for s in range(0, len(test), bs):
            batch = test[s:s + bs]
            obs_l, hid_l, pe_l, meta = [], [], [], []
            for u in batch:
                data = hf['data'][u].astype(np.float32)
                phase = hf['phase'][u].astype(np.float32)
                flags = hf['flags'][u].astype(np.float32)
                divisor = hf['dn_divisor'][u].astype(np.float32)
                np.random.seed(args.seed + int(u))     # fixed hole per unit -> comparable across ckpts/floors
                fm = fake_mask(flags, frac_range=args.frac_range, mode='mixed')
                if fm.sum() < 50:
                    continue
                hidden = np.clip(flags + fm, 0, 1)
                obs_l.append(np.stack([data, np.cos(phase), np.sin(phase)], 0))
                hid_l.append(hidden[None])
                if has_fpatch:
                    fmin = float(hf['freq_min_patch'][u]); fmax = float(hf['freq_max_patch'][u])
                else:
                    fmin, fmax = band_min, band_max
                key = (round(fmin, 3), round(fmax, 3))
                if key not in pe_cache:
                    pe_cache[key] = positional_encoding(fmin, fmax, band_min, band_max, sz, sz, cfg.pe_channels)
                pe_l.append(pe_cache[key])
                meta.append((data, phase, flags, divisor, fm))
            if not obs_l:
                continue
            x0 = torch.from_numpy(np.stack(obs_l, 0)).to(dev)
            m = torch.from_numpy(np.stack(hid_l, 0)).to(dev)
            pe_b = torch.from_numpy(np.stack(pe_l, 0).copy()).to(dev)
            cond = build_cond(x0, m, pe_b, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
            preds = {}
            for f, mk in zip(floors, mkeys):
                preds[mk] = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0,
                                        steps=args.steps, noise_floor=f).cpu().numpy()

            for i, (data, phase, flags, divisor, fm) in enumerate(meta):
                fmb = fm > 0.5
                rfb = flags > 0.5
                V_obs = (data * divisor * np.exp(1j * phase)).astype(np.complex128)
                # common reference: DPSS-fill the real-RFI channels (no truth there) so all variants
                # share it and the ONLY difference is the fake holes over known-good pixels.
                V_ref = V_obs.copy()
                if rfb.any():
                    V_ref[rfb] = dpss_fill(V_obs, rfb, A, args.dpss_lam)[rfb]
                V_dpss = V_ref.copy()                           # DPSS fits genuine good pixels (not RFI, not fake)
                V_dpss[fmb] = dpss_fill(V_obs, rfb | fmb, A, args.dpss_lam)[fmb]
                V_flag = V_ref.copy(); V_flag[fmb] = 0.0
                Pl['truth'].append(delay_power(V_ref, taper))   # true good values at the fake holes
                Pl['dpss'].append(delay_power(V_dpss, taper))
                Pl['flagged'].append(delay_power(V_flag, taper))
                for mk in mkeys:
                    amp_p = preds[mk][i, 0]; ph_p = np.arctan2(preds[mk][i, 2], preds[mk][i, 1])
                    Vm = V_ref.copy()
                    Vm[fmb] = (amp_p * divisor * np.exp(1j * ph_p))[fmb]
                    Pl[mk].append(delay_power(Vm, taper))
                n_acc += 1
            log(f"  {min(s + bs, len(test))}/{len(test)} batches  acc={n_acc}")

    hf.close()
    if n_acc == 0:
        log("no usable test tiles (all skipped for too-small fake holes); nothing to score")
        return
    Parr = {k: np.stack(Pl[k]) for k in variants}   # (n_tiles, sz)
    P = {k: Parr[k].mean(0) for k in variants}

    center = sz // 2
    hi = np.abs(np.arange(sz) - center) > args.fg_bins
    eps = 1e-30

    def wlogrmse_sp(sp, kk):
        w = sp['truth'] + eps
        d = (np.log10(sp[kk] + eps) - np.log10(sp['truth'] + eps)) ** 2
        return float(np.sqrt((w * d).sum() / w.sum()))

    def hiratio_sp(sp, kk):
        return float(sp[kk][hi].sum() / max(sp['truth'][hi].sum(), eps))

    def wlogrmse(kk):
        return wlogrmse_sp(P, kk)

    def hiratio(kk):
        return hiratio_sp(P, kk)

    log(f"=== FAKE-HOLE delay-space recovery on REAL held-out ({n_acc} tiles vs true good data) ===")
    log(f"  {'variant':<16}{'wlogP-RMSE':>12}{'hi-delay ratio':>16}")
    for kk in ['flagged', 'dpss'] + mkeys:
        log(f"  {kk:<16}{wlogrmse(kk):>12.4f}{hiratio(kk):>16.3f}")
    best_mk = min(mkeys, key=wlogrmse)                    # best noise_floor by headline metric

    # bootstrap over tiles: CI on the model-vs-DPSS wlogP-RMSE gap (positive = model better)
    brng = np.random.default_rng(args.seed)
    gaps = np.empty(args.bootstrap)
    for b in range(args.bootstrap):
        idx = brng.integers(0, n_acc, n_acc)
        sp = {k: Parr[k][idx].mean(0) for k in variants}
        gaps[b] = wlogrmse_sp(sp, 'dpss') - wlogrmse_sp(sp, best_mk)
    lo, himed = np.percentile(gaps, [2.5, 97.5])
    frac_win = float((gaps > 0).mean())
    log(f"  best model floor: {best_mk}")
    log(f"  bootstrap ({args.bootstrap}x over tiles): wlogP-RMSE gap (dpss - model) = "
        f"{gaps.mean():+.4f}  95% CI [{lo:+.4f}, {himed:+.4f}]  P(model<dpss)={frac_win:.3f}")
    win = wlogrmse(best_mk) < wlogrmse('dpss') and abs(hiratio(best_mk) - 1) < abs(hiratio('dpss') - 1)
    log(f"  verdict vs dpss: {'MODEL wins both' if win else 'split (model wins headline: '
        + str(wlogrmse(best_mk) < wlogrmse('dpss')) + ')'} "
        f"(wlogP-RMSE {wlogrmse(best_mk):.4f} vs dpss {wlogrmse('dpss'):.4f}; "
        f"hi-ratio {hiratio(best_mk):.3f} vs {hiratio('dpss'):.3f})")
    np.savez(args.out, **P, fg_bins=args.fg_bins, n=n_acc, floors=[str(f) for f in floors])
    log(f"saved power spectra -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-units', type=int, default=300, dest='max_units')
    ap.add_argument('--max-flag-frac', type=float, default=0.85, dest='max_flag_frac')
    ap.add_argument('--frac-range', type=float, nargs=2, default=(0.1, 0.25), dest='frac_range')
    ap.add_argument('--noise-floors', nargs='+', default=['none'], dest='noise_floors',
                    help="noise_floor values to sweep: none | auto | float (e.g. none 0.3 0.5 auto)")
    ap.add_argument('--dpss-hw', type=float, default=0.1, dest='dpss_hw')
    ap.add_argument('--dpss-lam', type=float, default=0.1, dest='dpss_lam')
    ap.add_argument('--fg-bins', type=int, default=20, dest='fg_bins')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
