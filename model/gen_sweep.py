import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, complex_mae, phase_error


def evaluate(diff, model, val_batch, cfg, dev):
    model.eval()
    x0 = val_batch['clean'].to(dev); m = val_batch['mask'].to(dev)
    cond = build_cond(val_batch['corrupted'].to(dev), m, val_batch['pe'].to(dev),
                      hole_fill=cfg.hole_fill)
    with torch.no_grad():
        pred = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0, steps=200)
    r = (m[:, 0] > 0).cpu().numpy()
    a_pred, a_true = pred[:, 0].cpu().numpy(), x0[:, 0].cpu().numpy()
    amp = float(np.abs(a_pred - a_true)[r].mean())
    mf = np.mean([np.abs(a_true[i][~r[i]].mean() - a_true[i])[r[i]].mean() for i in range(len(a_true))])
    cx = float(complex_mae(pred, x0, m))
    ph = float(phase_error(pred, x0, m))
    return amp, float(mf), cx, ph


def run_config(name, args, dev, rand_mask, time_roll, dropout):
    cfg = phase1(predict=args.predict)
    cfg.hole_fill = args.hole_fill
    print(f"\n{'='*60}\nCONFIG: {name}  (rand_mask={rand_mask} time_roll={time_roll} dropout={dropout})\n{'='*60}", flush=True)

    train_ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=True,
                            split='train', max_patches=args.n,
                            rand_mask=rand_mask, time_roll=time_roll)
    val_ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='val')
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=4)
    val_batch = {k: torch.stack([val_ds[i][k] for i in range(args.eval_n)]) for k in val_ds[0]}

    torch.manual_seed(0)
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size, dropout=dropout).to(dev)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    it = 0
    model.train()
    while it < args.iters:
        for batch in dl:
            opt.zero_grad()
            loss = diff.loss(model, {k: batch[k].to(dev) for k in batch}, cfg)
            loss.backward(); opt.step(); it += 1
            if it >= args.iters:
                break
        amp, mf, cx, ph = evaluate(diff, model, val_batch, cfg, dev)
        print(f"  it {it:5d}  loss {loss.item():.4f}  HELD-OUT: amp {amp:.4f} (mf {mf:.4f})  "
              f"complex {cx:.4f}  phase {ph:.3f}  {'BEATS' if amp < mf else 'loses'}", flush=True)
        model.train()


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  train_n={args.n}  eval on HELD-OUT val patches (real RFI mask)")
    run_config("baseline (no aug)",        args, dev, False, False, 0.0)
    run_config("rand_mask",                args, dev, True,  False, 0.0)
    run_config("time_roll",                args, dev, False, True,  0.0)
    run_config("dropout 0.1",              args, dev, False, False, 0.1)
    run_config("rand_mask+time_roll+drop", args, dev, True,  True,  0.1)
    print("\nDONE. Lowest HELD-OUT amp (and amp<mf) wins. That config generalises best.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--eval-n', type=int, default=32, dest='eval_n')
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--hole-fill', default='mean', dest='hole_fill')
    main(ap.parse_args())
