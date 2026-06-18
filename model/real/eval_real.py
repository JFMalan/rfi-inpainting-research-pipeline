import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase2
from data import RealDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, tre


def interp_baseline(obs, hidden):
    # per-row linear interp across freq through the holes, on the amplitude channel.
    # the honest recoverability bar (beats mean-fill ~2.5x on this structured data).
    out = obs.clone()
    a = obs[:, 0].cpu().numpy()
    h = (hidden[:, 0] > 0).cpu().numpy()
    nf = a.shape[-1]; idx = np.arange(nf)
    for i in range(a.shape[0]):
        for t in range(a.shape[1]):
            hr = h[i, t]
            if hr.any() and not hr.all():
                a[i, t, hr] = np.interp(idx[hr], idx[~hr], a[i, t, ~hr])
    out[:, 0] = torch.from_numpy(a).to(obs.device)
    return out


@torch.no_grad()
def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    cfg = phase2(predict=args.predict)
    print(f"device={dev} ({gpu})  ckpt={args.ckpt}  data={args.data}", flush=True)

    ds = RealDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='test',
                     fake_mask_frac=cfg.fake_mask_frac)
    print(f"test baselines: {len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)

    tres, fmaes, mf_maes, mf_tres, ip_maes = [], [], [], [], []
    seen = 0
    for batch in dl:
        obs = batch['obs'].to(dev); hidden = batch['hidden'].to(dev); fake = batch['fake_mask'].to(dev)
        cond = build_cond(obs, hidden, batch['pe'].to(dev), hole_fill=getattr(cfg, 'hole_fill', 'mean'))
        pred = diff.sample(model, cond, obs, hidden, predict=cfg.predict, eta=0.0, steps=200)
        tres.append(float(tre(pred, obs, fake))); fmaes.append(float(mae(pred, obs, fake)))
        base = obs.clone(); keep = hidden == 0
        for i in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                base[i, c] = obs[i, c][keep[i, 0]].mean()
        mf_maes.append(float(mae(base, obs, fake))); mf_tres.append(float(tre(base, obs, fake)))
        ip = interp_baseline(obs, hidden)
        ip_maes.append(float(mae(ip, obs, fake)))
        seen += obs.shape[0]
        if seen >= args.max_eval:
            break
    print(f"TEST RESULT  {args.tag}", flush=True)
    print(f"  n_eval {seen}  TRE model {np.mean(tres):.4f} mean-fill {np.mean(mf_tres):.4f}", flush=True)
    print(f"  fake-MAE  model {np.mean(fmaes):.4f}  interp {np.mean(ip_maes):.4f}  "
          f"mean-fill {np.mean(mf_maes):.4f}   (interp = the recoverable-structure bar)", flush=True)
    beats = "BEATS interp" if np.mean(fmaes) < np.mean(ip_maes) else (
            "beats mean-fill only" if np.mean(fmaes) < np.mean(mf_maes) else "ties/loses to mean-fill")
    print(f"  verdict: {beats}", flush=True)
    print(f"RESULTLINE\t{args.tag}\t{np.mean(tres):.4f}\t{np.mean(mf_tres):.4f}\t"
          f"{np.mean(fmaes):.4f}\t{np.mean(ip_maes):.4f}\t{np.mean(mf_maes):.4f}\t{seen}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-eval', type=int, default=64)
    main(ap.parse_args())
