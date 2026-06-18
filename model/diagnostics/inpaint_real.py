import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase2
from data import positional_encoding, build_cond
from diffusion import Diffusion
from unet import UNet


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    cfg = phase2(predict=args.predict)
    print(f"device={dev} ({gpu})  ckpt={args.ckpt}", flush=True)

    f = h5py.File(args.data, 'r')
    n_t = int(f.attrs['n_time']); n_f = int(f.attrs['n_freq'])
    band_min = float(f.attrs['freq_min_mhz']); band_max = float(f.attrs['freq_max_mhz'])
    ntot = f['data'].shape[0]
    rng = np.random.default_rng(args.seed)
    # prefer baselines with a moderate real-flag fraction (something to inpaint, but
    # enough context) so the demo is meaningful
    idxs = sorted(rng.choice(ntot, size=min(args.n, ntot), replace=False).tolist())

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    pe = positional_encoding(band_min, band_max, band_min, band_max, n_f, n_t, cfg.pe_channels)

    datas, phases, flagss, preds = [], [], [], []
    with torch.no_grad():
        for k, i in enumerate(idxs):
            data = f['data'][i].astype(np.float32)
            phase = f['phase'][i].astype(np.float32)
            flags = f['flags'][i].astype(np.float32)   # the REAL RFI mask = what we inpaint
            obs = np.stack([data, np.cos(phase), np.sin(phase)], 0)
            x0 = torch.from_numpy(obs)[None].to(dev)
            m = torch.from_numpy(flags)[None, None].to(dev)
            pe_t = torch.from_numpy(pe.copy())[None].to(dev)
            cond = build_cond(x0, m, pe_t, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
            pred = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0, steps=args.steps)
            datas.append(data); phases.append(phase); flagss.append(flags)
            preds.append(pred[0].cpu().numpy())
            print(f"  sampled {k+1}/{len(idxs)}", flush=True)
    f.close()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, data=np.array(datas), phase=np.array(phases),
             flags=np.array(flagss), pred=np.array(preds),
             band_min=band_min, band_max=band_max, idxs=np.array(idxs))
    print(f"saved npz -> {out}  (render with visualise_real_inpaint.py)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
