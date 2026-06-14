import argparse

import numpy as np
import torch

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, complex_mae


def run(diff, model, cond, x0, m, predict, clip, steps, eta=0.0):
    pred = diff.sample(model, cond, x0, m, predict=predict, clip=clip, eta=eta, steps=steps)
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

    # test deterministic (eta=0, smooth) vs stochastic (eta>0, adds texture) sampling.
    # also report error split by WIDE vs NARROW masked columns to expose the wide-hole failure.
    configs = [("eta0.0", 0.0), ("eta0.5", 0.5), ("eta1.0", 1.0)]
    saved = {}
    for name, eta in configs:
        amp, cx, pred = run(diff, model, cond, x0, m, cfg.predict, (-2.0, 4.0), 1000, eta=eta)
        # wide vs narrow: per masked column, width = contiguous masked extent along time
        wide_err, narrow_err = [], []
        for b in range(pred.shape[0]):
            mm = m[b, 0] > 0
            col_frac = mm.float().mean(dim=0)  # fraction masked per time column
            wide_cols = col_frac > 0.5
            ap, at = pred[b, 0], x0[b, 0]
            wide_px = mm & wide_cols[None, :]
            narrow_px = mm & ~wide_cols[None, :]
            if wide_px.any():
                wide_err.append((ap - at).abs()[wide_px].mean().item())
            if narrow_px.any():
                narrow_err.append((ap - at).abs()[narrow_px].mean().item())
        we = float(np.mean(wide_err)) if wide_err else 0.0
        ne = float(np.mean(narrow_err)) if narrow_err else 0.0
        print(f"  {name}:  amp_mae {amp:.4f}  complex {cx:.4f}  WIDE-mask MAE {we:.4f}  NARROW-mask MAE {ne:.4f}")
        saved[name] = pred.cpu().numpy()

    np.savez(args.out, clean=x0.cpu().numpy(), corrupted=batch['corrupted'].cpu().numpy(),
             mask=m.cpu().numpy(), pred_eta0=saved['eta0.0'],
             pred_eta05=saved['eta0.5'], pred_eta10=saved['eta1.0'],
             fmin=batch['fmin'].cpu().numpy(), fmax=batch['fmax'].cpu().numpy())
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6)
    main(ap.parse_args())
