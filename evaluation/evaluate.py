import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, psnr, phase_error


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase1()

    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split=args.split)
    print(f"{args.split} set: {len(ds)} patches  device={device}")
    if args.max_eval:
        ds.index = ds.index[:args.max_eval]
        print(f"evaluating on first {len(ds)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['ema'] if args.ema else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=device)

    maes, psnrs, pherrs = [], [], []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for bi, batch in enumerate(dl):
        x0 = batch['clean'].to(device)
        mask = batch['mask'].to(device)
        cond = build_cond(batch['corrupted'].to(device), mask, batch['pe'].to(device))
        pred = diff.sample(model, cond, x0, mask, predict=cfg.predict)
        maes.append(float(mae(pred, x0, mask)))
        psnrs.append(float(psnr(pred, x0, mask)))
        pherrs.append(float(phase_error(pred, x0, mask)))
        if saved < args.save_batches:
            np.savez(out / f'eval_b{bi}.npz', clean=x0.cpu().numpy(),
                     corrupted=batch['corrupted'].numpy(), mask=batch['mask'].numpy(),
                     pred=pred.cpu().numpy(),
                     fmin=batch['fmin'].numpy(), fmax=batch['fmax'].numpy())
            saved += 1
        print(f"batch {bi+1}/{len(dl)}  mae={maes[-1]:.4f}  psnr={psnrs[-1]:.3f}  "
              f"phase_err={pherrs[-1]:.4f}", flush=True)

    res = {'split': args.split, 'n_batches': len(maes),
           'mae_mean': float(np.mean(maes)), 'mae_std': float(np.std(maes)),
           'psnr_mean': float(np.mean(psnrs)), 'psnr_std': float(np.std(psnrs)),
           'phase_err_mean': float(np.mean(pherrs)), 'phase_err_std': float(np.std(pherrs)),
           'ckpt': args.ckpt, 'ema': args.ema}
    (out / 'metrics.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-eval', type=int, default=None)
    ap.add_argument('--save-batches', type=int, default=4)
    ap.add_argument('--ema', action='store_true', default=True)
    main(ap.parse_args())
