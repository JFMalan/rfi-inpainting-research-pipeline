import argparse

import numpy as np
import torch

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, complex_mae


def run(diff, model, cond, x0, m, predict, clip, steps):
    pred = diff.sample(model, cond, x0, m, predict=predict, clip=clip, eta=0.0, steps=steps)
    return float(mae(pred, x0, m)), float(complex_mae(pred, x0, m)), pred


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase1()
    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='val')
    n = args.n
    batch = {k: torch.stack([ds[i][k] for i in range(n)]).to(dev) for k in ds[0]}
    x0, m = batch['clean'], batch['mask']
    cond = build_cond(batch['corrupted'], m, batch['pe'], hole_fill=cfg.hole_fill)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema']); model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)

    # data-driven tight clamp from the known region
    known = x0[:, 0][m[:, 0] == 0]
    lo, hi = float(known.mean() - 5 * known.std()), float(known.mean() + 5 * known.std())
    print(f"data range (known amp): mean {known.mean():.3f} std {known.std():.3f}  tight clip ({lo:.2f},{hi:.2f})")

    configs = [
        ("current  (200 steps, clip -2..4)", (-2.0, 4.0), 200),
        ("full     (1000 steps, clip -2..4)", (-2.0, 4.0), 1000),
        ("tight    (1000 steps, clip tight)", (lo, hi),    1000),
    ]
    saved = {}
    for name, clip, steps in configs:
        amp, cx, pred = run(diff, model, cond, x0, m, cfg.predict, clip, steps)
        oh = pred[:, 0][m[:, 0] > 0]
        print(f"  {name}:  amp_mae {amp:.4f}  complex {cx:.4f}  pred_hole[{oh.min():.2f},{oh.max():.2f}]")
        saved[name.split()[0]] = pred.cpu().numpy()

    np.savez(args.out, clean=x0.cpu().numpy(), corrupted=batch['corrupted'].cpu().numpy(),
             mask=m.cpu().numpy(), pred_current=saved['current'],
             pred_full=saved['full'], pred_tight=saved['tight'],
             fmin=batch['fmin'].cpu().numpy(), fmax=batch['fmax'].cpu().numpy())
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6)
    main(ap.parse_args())
