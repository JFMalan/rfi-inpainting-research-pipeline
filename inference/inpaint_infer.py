import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))

from config import phase1, phase2
from data import positional_encoding, build_cond, smooth_component
from diffusion import Diffusion
from unet import UNet

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    amp_key = 'corrupted' if args.sim else 'data'
    hole_key = 'mask' if args.sim else 'flags'
    nf = None if args.noise_floor in (None, 'none') else (
        'auto' if args.noise_floor == 'auto' else float(args.noise_floor))
    log(f"device={dev} ({gpu})  h5={args.h5}  ckpt={args.ckpt}  noise_floor={nf}")

    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    band_min = float(hf.attrs['freq_min_mhz'])
    band_max = float(hf.attrs['freq_max_mhz'])
    n_units = hf[amp_key].shape[0]
    cap = n_units if args.max_units is None else min(args.max_units, n_units)

    if args.oracle:
        true_key = 'clean' if args.sim else amp_key
        log(f"ORACLE mode: writing TRUE {true_key} (no model) to isolate the write-back path")
        preds = np.empty((cap, 3, sz, sz), dtype=np.float32)
        for u in range(cap):
            ph = hf['phase'][u].astype(np.float32)
            preds[u] = np.stack([hf[true_key][u].astype(np.float32), np.cos(ph), np.sin(ph)], 0)
        hf.close()
        np.savez(args.out_preds, preds=preds)
        log(f"saved {cap} oracle preds {preds.shape} -> {args.out_preds}")
        return

    cfg = (phase1 if args.sim else phase2)(predict=args.predict)
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    has_fpatch = 'freq_min_patch' in hf
    pe_cache = {}

    def get_pe(u):
        if has_fpatch:
            fmin = float(hf['freq_min_patch'][u]); fmax = float(hf['freq_max_patch'][u])
        else:
            fmin, fmax = band_min, band_max
        key = (round(fmin, 3), round(fmax, 3))
        pe = pe_cache.get(key)
        if pe is None:
            pe = positional_encoding(fmin, fmax, band_min, band_max, sz, sz, cfg.pe_channels)
            pe_cache[key] = pe
        return pe

    log(f"model loaded; inferring {cap}/{n_units} units")

    bs = args.batch
    preds = np.empty((cap, 3, sz, sz), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, cap, bs):
            e = min(s + bs, cap)
            obs_l, hole_l, pe_l = [], [], []
            for u in range(s, e):
                data = hf[amp_key][u].astype(np.float32)
                phase = hf['phase'][u].astype(np.float32)
                hole = hf[hole_key][u].astype(np.float32)
                amp = smooth_component(data, hole, args.smooth_sigma) if args.smooth_target else data
                obs_l.append(np.stack([amp, np.cos(phase), np.sin(phase)], 0))
                hole_l.append(hole[None])
                pe_l.append(get_pe(u))
            x0 = torch.from_numpy(np.stack(obs_l, 0)).to(dev)
            m = torch.from_numpy(np.stack(hole_l, 0)).to(dev)
            pe_b = torch.from_numpy(np.stack(pe_l, 0).copy()).to(dev)
            cond = build_cond(x0, m, pe_b, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
            pred = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0,
                               steps=args.steps, noise_floor=nf)
            preds[s:e] = pred.cpu().numpy()
            log(f"  inferred {e}/{cap}  ({e / max(time.time() - t0, 1e-6):.2f} units/s)")
    hf.close()
    np.savez(args.out_preds, preds=preds)
    log(f"saved {cap} preds {preds.shape} -> {args.out_preds}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-preds', required=True, dest='out_preds')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--noise-floor', default='auto', dest='noise_floor')
    ap.add_argument('--smooth-target', action='store_true', dest='smooth_target')
    ap.add_argument('--smooth-sigma', type=float, default=1.0, dest='smooth_sigma')
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    ap.add_argument('--oracle', action='store_true')
    main(ap.parse_args())
