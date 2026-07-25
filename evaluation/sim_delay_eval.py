import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import phase1
from data import positional_encoding, build_cond, fake_mask
from diffusion import Diffusion
from unet import UNet
from classical_fill import dpss_basis, dpss_fill, gpr_fill

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

    ck = torch.load(args.ckpt, map_location=dev)
    cfg = phase1(**ck['cfg']) if 'cfg' in ck else phase1()
    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['n_freq'])
    band_min = float(hf.attrs['freq_min_mhz'])
    band_max = float(hf.attrs['freq_max_mhz'])
    n = hf['clean'].shape[0]
    idx_all = np.arange(n)
    if args.max_units:
        rng = np.random.default_rng(0)
        idx_all = np.sort(rng.choice(n, size=min(args.max_units, n), replace=False))
    log(f"device={dev} ({gpu})  h5={args.h5}  units={len(idx_all)}  clean_target eval vs noise-free truth")

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    A = dpss_basis(sz, args.dpss_hw)
    taper = blackman_harris(sz)

    floors = [None if f in ('none', 'None') else ('auto' if f == 'auto' else float(f))
              for f in args.noise_floors]
    mkeys = [f"model_nf{('none' if f is None else f)}" for f in floors]
    variants = ['truth', 'dpss', 'gpr', 'flagged'] + mkeys
    Pl = {k: [] for k in variants}
    pe_cache = {}
    bs = args.batch
    n_acc = 0
    with torch.no_grad():
        for s in range(0, len(idx_all), bs):
            batch = idx_all[s:s + bs]
            obs_l, hid_l, pe_l, meta = [], [], [], []
            for u in batch:
                clean = hf['clean'][u].astype(np.float32)        # noisy RFI-free amplitude (observation)
                phase = hf['phase'][u].astype(np.float32)
                amp_t = hf['amp_target'][u].astype(np.float32)   # noise-free truth amplitude
                phase_t = hf['phase_target'][u].astype(np.float32)
                rfi = hf['mask'][u].astype(np.float32)
                divisor = hf['dn_divisor'][u].astype(np.float32)
                np.random.seed(args.seed + int(u))
                fm = fake_mask(rfi, frac_range=args.frac_range, mode='mixed')
                if fm.sum() < 50:
                    continue
                hidden = np.clip(rfi + fm, 0, 1)
                obs_l.append(np.stack([clean, np.cos(phase), np.sin(phase)], 0))
                hid_l.append(hidden[None])
                if 'freq_min_patch' in hf:
                    fmin = float(hf['freq_min_patch'][u]); fmax = float(hf['freq_max_patch'][u])
                else:
                    fmin, fmax = band_min, band_max
                key = (round(fmin, 3), round(fmax, 3))
                if key not in pe_cache:
                    pe_cache[key] = positional_encoding(fmin, fmax, band_min, band_max, sz, sz, cfg.pe_channels)
                pe_l.append(pe_cache[key])
                meta.append((clean, phase, amp_t, phase_t, rfi, divisor, fm))
            if not obs_l:
                continue
            x0 = torch.from_numpy(np.stack(obs_l, 0)).to(dev)
            m = torch.from_numpy(np.stack(hid_l, 0)).to(dev)
            pe_b = torch.from_numpy(np.stack(pe_l, 0).copy()).to(dev)
            cond = build_cond(x0, m, pe_b, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
            preds = {mk: diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0,
                                     steps=args.steps, noise_floor=f).cpu().numpy()
                     for f, mk in zip(floors, mkeys)}

            for i, (clean, phase, amp_t, phase_t, rfi, divisor, fm) in enumerate(meta):
                fmb = fm > 0.5
                gap = (rfi > 0.5) | fmb
                V_true = (amp_t * divisor * np.exp(1j * phase_t)).astype(np.complex128)   # noise-free truth
                V_obs = (clean * divisor * np.exp(1j * phase)).astype(np.complex128)      # noisy observation
                V_dpss = V_true.copy(); V_dpss[fmb] = dpss_fill(V_obs, gap, A, args.dpss_lam)[fmb]
                V_gpr = V_true.copy(); V_gpr[fmb] = gpr_fill(V_obs, gap, args.gpr_ell, args.gpr_noise)[fmb]
                V_flag = V_true.copy(); V_flag[fmb] = 0.0
                Pl['truth'].append(delay_power(V_true, taper))
                Pl['dpss'].append(delay_power(V_dpss, taper))
                Pl['gpr'].append(delay_power(V_gpr, taper))
                Pl['flagged'].append(delay_power(V_flag, taper))
                for mk in mkeys:
                    amp_p = preds[mk][i, 0]; ph_p = np.arctan2(preds[mk][i, 2], preds[mk][i, 1])
                    Vm = V_true.copy(); Vm[fmb] = (amp_p * divisor * np.exp(1j * ph_p))[fmb]
                    Pl[mk].append(delay_power(Vm, taper))
                n_acc += 1
            log(f"  {min(s + bs, len(idx_all))}/{len(idx_all)}  acc={n_acc}")

    hf.close()
    if n_acc == 0:
        log("no usable tiles"); return
    Parr = {k: np.stack(Pl[k]) for k in variants}
    P = {k: Parr[k].mean(0) for k in variants}
    center = sz // 2
    hi = np.abs(np.arange(sz) - center) > args.fg_bins
    eps = 1e-30

    def wlog(kk):
        w = P['truth'] + eps
        d = (np.log10(P[kk] + eps) - np.log10(P['truth'] + eps)) ** 2
        return float(np.sqrt((w * d).sum() / w.sum()))

    def hiratio(kk):
        return float(P[kk][hi].sum() / max(P['truth'][hi].sum(), eps))

    log(f"=== R4 SIM delay recovery vs noise-free truth ({n_acc} tiles), R3 ckpt + sampling sweep ===")
    log(f"  {'variant':<16}{'wlogP-RMSE':>12}{'hi-ratio':>10}")
    for kk in ['flagged', 'dpss', 'gpr'] + mkeys:
        log(f"  {kk:<16}{wlog(kk):>12.4f}{hiratio(kk):>10.3f}")
    best_mk = min(mkeys, key=wlog)
    classical = min(['dpss', 'gpr'], key=wlog)
    log(f"  best sampling: {best_mk}  |  stronger classical: {classical}  "
        f"(model {wlog(best_mk):.4f} vs {classical} {wlog(classical):.4f})")
    np.savez(args.out, **{f'P_{k}': P[k] for k in variants},
             **{f'tiles_{k}': Parr[k] for k in variants}, fg_bins=args.fg_bins, n=n_acc)
    log(f"saved -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-units', type=int, default=300, dest='max_units')
    ap.add_argument('--frac-range', type=float, nargs=2, default=(0.1, 0.25), dest='frac_range')
    ap.add_argument('--noise-floors', nargs='+', default=['none', '0.3', '0.5', 'auto'], dest='noise_floors')
    ap.add_argument('--dpss-hw', type=float, default=0.1, dest='dpss_hw')
    ap.add_argument('--dpss-lam', type=float, default=0.1, dest='dpss_lam')
    ap.add_argument('--gpr-ell', type=float, default=30.0, dest='gpr_ell')
    ap.add_argument('--gpr-noise', type=float, default=0.05, dest='gpr_noise')
    ap.add_argument('--fg-bins', type=int, default=20, dest='fg_bins')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
