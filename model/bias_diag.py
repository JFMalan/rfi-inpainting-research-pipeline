import argparse

import numpy as np
import torch

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet


def band_geometry(mask_2d):
    # mask_2d (T, F): RFI bands are masked frequency columns. Per masked pixel,
    # distance (in freq bins) to the nearest unmasked column in its row, and the
    # contiguous band width it belongs to.
    T, F = mask_2d.shape
    depth = np.full((T, F), -1, dtype=np.int32)
    width = np.zeros((T, F), dtype=np.int32)
    for t in range(T):
        row = mask_2d[t] > 0
        if not row.any():
            continue
        f = 0
        while f < F:
            if row[f]:
                j = f
                while j < F and row[j]:
                    j += 1
                w = j - f
                for k in range(f, j):
                    width[t, k] = w
                    dl = k - (f - 1) if f > 0 else 10 ** 6
                    dr = j - k if j < F else 10 ** 6
                    depth[t, k] = min(dl, dr)
                f = j
            else:
                f += 1
    return depth, width


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase1(predict=args.predict)
    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='val')
    n = args.n
    batch = {k: torch.stack([ds[i][k] for i in range(n)]).to(dev) for k in ds[0]}
    x0, m = batch['clean'], batch['mask']
    cond = build_cond(batch['corrupted'], m, batch['pe'], hole_fill=cfg.hole_fill)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)

    pred = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0, steps=args.steps)

    ap = pred[:, 0].cpu().numpy()
    at = x0[:, 0].cpu().numpy()
    mk = m[:, 0].cpu().numpy()

    all_signed, all_depth, all_width, all_r2 = [], [], [], []
    for b in range(n):
        depth, width = band_geometry(mk[b])
        hole = mk[b] > 0
        if hole.sum() < 5:
            continue
        # per-patch recoverability proxy: variance of known amplitude (structure -> high)
        known = at[b][~hole]
        struct = float(known.std())
        signed = (ap[b] - at[b])[hole]
        all_signed.append(signed)
        all_depth.append(depth[hole])
        all_width.append(width[hole])
        all_r2.append(np.full(hole.sum(), struct))

    signed = np.concatenate(all_signed)
    depth = np.concatenate(all_depth)
    width = np.concatenate(all_width)
    struct = np.concatenate(all_r2)

    print(f"predict={cfg.predict}  patches={n}  hole pixels={signed.size}")
    print(f"overall signed error (bias): mean {signed.mean():+.4f}  abs {np.abs(signed).mean():.4f}")
    print("\nsigned error vs DEPTH into band (distance to nearest known freq column):")
    print(f"{'depth':>8} {'n':>7} {'mean_signed':>12} {'mean_abs':>10}")
    for lo, hi in [(1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6)]:
        sel = (depth >= lo) & (depth <= hi)
        if sel.sum():
            print(f"{lo:>3}-{hi if hi < 100 else '+':<4} {sel.sum():>7} "
                  f"{signed[sel].mean():>+12.4f} {np.abs(signed[sel]).mean():>10.4f}")

    print("\nsigned error vs BAND WIDTH:")
    print(f"{'width':>8} {'n':>7} {'mean_signed':>12} {'mean_abs':>10}")
    for lo, hi in [(1, 4), (5, 12), (13, 30), (31, 10 ** 6)]:
        sel = (width >= lo) & (width <= hi)
        if sel.sum():
            print(f"{lo:>3}-{hi if hi < 1000 else '+':<4} {sel.sum():>7} "
                  f"{signed[sel].mean():>+12.4f} {np.abs(signed[sel]).mean():>10.4f}")

    print("\nsigned error vs per-patch STRUCTURE (known-region std; low=noise-dominated):")
    print(f"{'struct':>10} {'n':>7} {'mean_signed':>12} {'mean_abs':>10}")
    qs = np.quantile(struct, [0, 0.33, 0.66, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (struct >= lo) & (struct <= hi)
        if sel.sum():
            print(f"{lo:>10.3f} {sel.sum():>7} {signed[sel].mean():>+12.4f} "
                  f"{np.abs(signed[sel]).mean():>10.4f}  (<= {hi:.3f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--predict', default='x0', choices=['noise', 'x0'])
    ap.add_argument('--steps', type=int, default=200)
    main(ap.parse_args())
