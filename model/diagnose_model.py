import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet


@torch.no_grad()
def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase1()
    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='val')
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)

    batch = next(iter(dl))
    x0 = batch['clean'].to(dev)
    m = batch['mask'].to(dev)
    cond = build_cond(batch['corrupted'].to(dev), m, batch['pe'].to(dev))

    # noise-prediction error inside vs outside mask, per channel, averaged over timesteps
    print("noise-prediction |pred-eps| by channel (amp/cos/sin), in-mask vs out-mask:")
    for t_val in [50, 200, 500, 800]:
        t = torch.full((x0.shape[0],), t_val, device=dev, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = diff.q_sample(x0, t, eps)
        pred = model(torch.cat([xt, cond], dim=1), t)
        err = (pred - eps).abs()
        inm = m > 0
        out = m == 0
        line = [f"t={t_val:4d}"]
        for c in range(x0.shape[1]):
            ein = err[:, c:c+1][inm].mean().item()
            eout = err[:, c:c+1][out].mean().item()
            line.append(f"ch{c}: in={ein:.3f} out={eout:.3f}")
        print("  " + "  ".join(line))

    # one-step x0 reconstruction from a mid-level noised state
    t = torch.full((x0.shape[0],), 200, device=dev, dtype=torch.long)
    eps = torch.randn_like(x0)
    xt = diff.q_sample(x0, t, eps)
    x0_pred, _ = diff.predict_x0(model, xt, cond, t, clip=(-2, 4))
    inm = m > 0
    amp_err_in = (x0_pred[:, 0:1] - x0[:, 0:1]).abs()[inm].mean().item()
    print(f"\none-step x0 amplitude error in-mask (t=200): {amp_err_in:.4f}")
    print(f"clean amp std: {x0[:,0].std().item():.4f}  (mean-fill would give ~this)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    main(ap.parse_args())
