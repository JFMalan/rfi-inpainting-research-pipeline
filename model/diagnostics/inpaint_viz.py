import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase1, phase2
from data import positional_encoding, fake_mask, build_cond
from diffusion import Diffusion
from unet import UNet


def load_sim(f, idx, pe_ch, band_min, band_max, n_t, n_f):
    clean = f['clean'][idx].astype(np.float32)
    corrupted = f['corrupted'][idx].astype(np.float32)
    mask = f['mask'][idx].astype(np.float32)
    phase = f['phase'][idx].astype(np.float32)
    cos_p, sin_p = np.cos(phase), np.sin(phase)
    x0 = np.stack([clean, cos_p, sin_p], 0)
    cond_src = np.stack([corrupted, cos_p, sin_p], 0)
    pe = positional_encoding(band_min, band_max, band_min, band_max, n_f, n_t, pe_ch)
    return x0, cond_src, mask, pe


def load_real(f, idx, pe_ch, band_min, band_max, n_t, n_f):
    data = f['data'][idx].astype(np.float32)
    phase = f['phase'][idx].astype(np.float32)
    real_flags = f['flags'][idx].astype(np.float32)
    fm = fake_mask(real_flags)
    cos_p, sin_p = np.cos(phase), np.sin(phase)
    obs = np.stack([data, cos_p, sin_p], 0)
    hidden = np.clip(real_flags + fm, 0, 1)
    pe = positional_encoding(band_min, band_max, band_min, band_max, n_f, n_t, pe_ch)
    # for real: "clean" shown = observed (no GT); mask shown = the fake holes we score;
    # conditioning hides real flags + fake holes
    return obs, obs, hidden, fm, pe


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    cfg = (phase2 if args.real else phase1)(predict=args.predict)
    print(f"device={dev} ({gpu})  {'REAL' if args.real else 'SIM'} data  ckpt={args.ckpt}", flush=True)

    f = h5py.File(args.data, 'r')
    n_t = int(f.attrs['n_time']); n_f = int(f.attrs['n_freq'])
    band_min = float(f.attrs['freq_min_mhz']); band_max = float(f.attrs['freq_max_mhz'])
    ntot = f['data'].shape[0] if args.real else f['clean'].shape[0]
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(ntot, size=min(args.n, ntot), replace=False).tolist())

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)

    cleans, corrs, masks, preds = [], [], [], []
    with torch.no_grad():
        for k, i in enumerate(idxs):
            if args.real:
                x0_np, cond_np, hidden_np, fake_np, pe_np = load_real(f, i, cfg.pe_channels, band_min, band_max, n_t, n_f)
                show_mask = fake_np
                cond_mask = hidden_np
            else:
                x0_np, cond_np, mask_np, pe_np = load_sim(f, i, cfg.pe_channels, band_min, band_max, n_t, n_f)
                show_mask = mask_np
                cond_mask = mask_np
            x0 = torch.from_numpy(x0_np)[None].to(dev)
            cond_src = torch.from_numpy(cond_np)[None].to(dev)
            cmask = torch.from_numpy(cond_mask)[None, None].to(dev)
            pe = torch.from_numpy(pe_np.copy())[None].to(dev)
            cond = build_cond(cond_src, cmask, pe, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
            pred = diff.sample(model, cond, x0, cmask, predict=cfg.predict, eta=0.0, steps=args.steps)
            cleans.append(x0[0].cpu().numpy())
            corrs.append(cond_src[0].cpu().numpy())
            masks.append(show_mask)
            preds.append(pred[0].cpu().numpy())
            print(f"  sampled {k+1}/{len(idxs)}", flush=True)
    f.close()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             clean=np.array(cleans), corrupted=np.array(corrs),
             mask=np.array(masks)[:, None], pred=np.array(preds),
             fmin=np.full(len(idxs), band_min, np.float32),
             fmax=np.full(len(idxs), band_max, np.float32))
    print(f"saved npz -> {out}  (render with visualise_samples.py)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
