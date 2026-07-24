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
from data import positional_encoding, build_cond
from diffusion import Diffusion
from unet import UNet
from classical_fill import dpss_basis, dpss_fill, gpr_fill, clean_fill, lssa_fill

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


def shift_flags(flags, shift):
    sm = np.roll(flags > 0.5, shift, axis=1)
    return np.clip(sm.astype(np.float32) - (flags > 0.5), 0, 1)


def dphi(a, b):
    d = np.mod(a - b, 2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def wide_mask(fm, thresh):
    # per-pixel: True where the fake hole belongs to a run of >= thresh consecutive
    # flagged frequency channels (wideband RFI); else narrowband.
    out = np.zeros_like(fm, dtype=bool)
    for t in range(fm.shape[0]):
        row = fm[t] > 0.5
        i = 0
        while i < len(row):
            if row[i]:
                j = i
                while j < len(row) and row[j]:
                    j += 1
                if j - i >= thresh:
                    out[t, i:j] = True
                i = j
            else:
                i += 1
    return out


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'

    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    band_min = float(hf.attrs['freq_min_mhz'])
    band_max = float(hf.attrs['freq_max_mhz'])
    n = hf['data'].shape[0]
    split = hf['split'][:] if 'split' in hf else np.ones(n, np.int32)
    has_fpatch = 'freq_min_patch' in hf

    flags_frac = hf['flags'][:].astype(np.float32).mean(axis=(1, 2))
    test = np.where((split == 1) & (flags_frac < args.max_flag_frac))[0]
    if args.max_units:
        test = test[:args.max_units]
    log(f"device={dev} ({gpu})  h5={args.h5}")
    log(f"test units={len(test)} (split==1, flagfrac<{args.max_flag_frac})  shift={args.shift} chan")

    cfg = phase2(predict=args.predict)
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    A = dpss_basis(sz, args.dpss_hw)
    taper = blackman_harris(sz)
    log(f"model loaded; ckpt={args.ckpt}")

    floors = [None if f in ('none', 'None') else ('auto' if f == 'auto' else float(f))
              for f in args.noise_floors]
    mkeys = [f"model_nf{('none' if f is None else f)}" for f in floors]
    methods = ['dpss', 'gpr', 'clean', 'lssa', 'flagged'] + mkeys
    variants = ['truth'] + methods

    Pl = {k: [] for k in variants}                       # per-tile delay power
    af = {k: {'nb': [], 'wb': []} for k in methods}       # amplitude fractional error samples
    ph = {k: {'nb': [], 'wb': []} for k in methods}       # phase error samples (rad)

    pe_cache = {}
    bs = args.batch
    n_acc = 0
    with torch.no_grad():
        for s in range(0, len(test), bs):
            batch = test[s:s + bs]
            obs_l, hid_l, pe_l, meta = [], [], [], []
            for u in batch:
                data = hf['data'][u].astype(np.float32)
                phase = hf['phase'][u].astype(np.float32)
                flags = hf['flags'][u].astype(np.float32)
                divisor = hf['dn_divisor'][u].astype(np.float32)
                fm = shift_flags(flags, args.shift)
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
            preds = {mk: diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0,
                                     steps=args.steps, noise_floor=f).cpu().numpy()
                     for f, mk in zip(floors, mkeys)}

            for i, (data, phase, flags, divisor, fm) in enumerate(meta):
                fmb = fm > 0.5
                rfb = flags > 0.5
                wb = wide_mask(fm, args.wide_thresh) & fmb
                nb = fmb & ~wb
                V_obs = (data * divisor * np.exp(1j * phase)).astype(np.complex128)
                V_ref = V_obs.copy()
                if rfb.any():
                    V_ref[rfb] = dpss_fill(V_obs, rfb, A, args.dpss_lam)[rfb]   # real RFI -> reference fill
                amp_t = np.abs(V_ref); ph_t = np.angle(V_ref)

                fills = {}
                V_dpss = V_ref.copy(); V_dpss[fmb] = dpss_fill(V_obs, rfb | fmb, A, args.dpss_lam)[fmb]
                fills['dpss'] = V_dpss
                V_gpr = V_ref.copy(); V_gpr[fmb] = gpr_fill(V_obs, rfb | fmb, args.gpr_ell, args.gpr_noise)[fmb]
                fills['gpr'] = V_gpr
                V_clean = V_ref.copy(); V_clean[fmb] = clean_fill(V_obs, rfb | fmb, taper)[fmb]
                fills['clean'] = V_clean
                V_lssa = V_ref.copy(); V_lssa[fmb] = lssa_fill(V_obs, rfb | fmb, args.lssa_nmax)[fmb]
                fills['lssa'] = V_lssa
                V_flag = V_ref.copy(); V_flag[fmb] = 0.0
                fills['flagged'] = V_flag
                for mk in mkeys:
                    amp_p = preds[mk][i, 0]; ph_p = np.arctan2(preds[mk][i, 2], preds[mk][i, 1])
                    Vm = V_ref.copy(); Vm[fmb] = (amp_p * divisor * np.exp(1j * ph_p))[fmb]
                    fills[mk] = Vm

                Pl['truth'].append(delay_power(V_ref, taper))
                for k, Vk in fills.items():
                    Pl[k].append(delay_power(Vk, taper))
                    dv = (np.abs(Vk) - amp_t) / np.maximum(amp_t, 1e-12)   # eq 17 (amplitude)
                    dp = dphi(np.angle(Vk), ph_t)                          # eq 18 (phase)
                    for reg, sel in (('nb', nb), ('wb', wb)):
                        if sel.any():
                            af[k][reg].append(dv[sel]); ph[k][reg].append(dp[sel])
                n_acc += 1
            log(f"  {min(s + bs, len(test))}/{len(test)}  acc={n_acc}")

    hf.close()
    if n_acc == 0:
        log("no usable tiles"); return

    Parr = {k: np.stack(Pl[k]) for k in variants}
    P = {k: Parr[k].mean(0) for k in variants}
    center = sz // 2
    tau = np.arange(sz) - center
    inw = np.abs(tau) <= args.fg_bins
    eps = 1e-30

    def pfrac(kk, sel):
        return float(((P[kk][sel] - P['truth'][sel]) / (P['truth'][sel] + eps)).mean())

    def wlog(kk):
        w = P['truth'] + eps
        d = (np.log10(P[kk] + eps) - np.log10(P['truth'] + eps)) ** 2
        return float(np.sqrt((w * d).sum() / w.sum()))

    log(f"=== PAGANO-STYLE real eval: shifted-flag holes ({n_acc} tiles), delay-selected model ===")
    log(f"  {'method':<14}{'|dV|frac nb':>13}{'|dV|frac wb':>13}{'dphi nb':>10}{'dphi wb':>10}"
        f"{'dP/P in':>10}{'dP/P out':>10}{'wlogP':>9}")
    stats = {}
    for k in methods:
        row = {}
        for reg in ('nb', 'wb'):
            a = np.concatenate(af[k][reg]) if af[k][reg] else np.array([np.nan])
            p = np.concatenate(ph[k][reg]) if ph[k][reg] else np.array([np.nan])
            row[f'af_{reg}_std'] = float(np.nanstd(a)); row[f'af_{reg}_med'] = float(np.nanmedian(np.abs(a)))
            row[f'ph_{reg}_med'] = float(np.nanmedian(p))
        row['dP_in'] = pfrac(k, inw); row['dP_out'] = pfrac(k, ~inw); row['wlog'] = wlog(k)
        stats[k] = row
        log(f"  {k:<14}{row['af_nb_med']:>13.4f}{row['af_wb_med']:>13.4f}"
            f"{row['ph_nb_med']:>10.4f}{row['ph_wb_med']:>10.4f}"
            f"{row['dP_in']:>10.4f}{row['dP_out']:>10.4f}{row['wlog']:>9.4f}")
    log("  columns: median |amplitude frac err| and median phase err (rad) for narrowband/wideband")
    log("           holes; mean fractional power-spectrum error inside/outside the wedge; wlogP-RMSE")

    best_mk = min(mkeys, key=wlog)
    classical = min(['dpss', 'gpr', 'clean', 'lssa'], key=wlog)
    log(f"  best model={best_mk}  stronger classical={classical}")
    log(f"  model vs {classical}: amp(wb) {stats[best_mk]['af_wb_med']:.4f} vs {stats[classical]['af_wb_med']:.4f}; "
        f"phase(wb) {stats[best_mk]['ph_wb_med']:.4f} vs {stats[classical]['ph_wb_med']:.4f}; "
        f"dP_out {stats[best_mk]['dP_out']:.4f} vs {stats[classical]['dP_out']:.4f}")
    np.savez(args.out, **{f'P_{k}': P[k] for k in variants},
             **{f'tiles_{k}': Parr[k] for k in variants},
             fg_bins=args.fg_bins, n=n_acc, shift=args.shift,
             stats=np.array([str(stats)]))
    log(f"saved -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-units', type=int, default=400, dest='max_units')
    ap.add_argument('--max-flag-frac', type=float, default=0.85, dest='max_flag_frac')
    ap.add_argument('--shift', type=int, default=40, help='frequency-channel shift for the fake-flag mask')
    ap.add_argument('--wide-thresh', type=int, default=5, dest='wide_thresh',
                    help='consecutive flagged channels to count a hole as wideband')
    ap.add_argument('--noise-floors', nargs='+', default=['none', '0.5'], dest='noise_floors')
    ap.add_argument('--dpss-hw', type=float, default=0.1, dest='dpss_hw')
    ap.add_argument('--dpss-lam', type=float, default=0.1, dest='dpss_lam')
    ap.add_argument('--gpr-ell', type=float, default=30.0, dest='gpr_ell')
    ap.add_argument('--gpr-noise', type=float, default=0.05, dest='gpr_noise')
    ap.add_argument('--lssa-nmax', type=int, default=32, dest='lssa_nmax')
    ap.add_argument('--fg-bins', type=int, default=20, dest='fg_bins')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
