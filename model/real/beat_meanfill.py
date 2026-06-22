import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase2
from data import RealDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, complex_mae, phase_error


def meanfill(obs, hidden):
    base = obs.clone()
    keep = hidden == 0
    for i in range(obs.shape[0]):
        km = keep[i, 0]
        for c in range(obs.shape[1]):
            base[i, c][~km] = obs[i, c][km].mean()
    return base


AREA_EDGES = [0, 64, 256, 1024, np.inf]   # px; ~<8, 8-16, 16-32, >32 effective diameter


def stratify_amp(pred, obs, fake, hidden, acc):
    # per connected hole component: |amp err| for model vs mean-fill, bucketed by area.
    # mean-fill value = mean over genuinely observed pixels (hidden==0): exclude BOTH the
    # fake holes and the real RFI flags, else the flagged RFI tail contaminates the mean.
    p = pred[:, 0].cpu().numpy(); o = obs[:, 0].cpu().numpy()
    fk = (fake[:, 0] > 0).cpu().numpy()
    keepobs = (hidden[:, 0] == 0).cpu().numpy()
    for i in range(p.shape[0]):
        lab, n = label(fk[i])
        if n == 0:
            continue
        mf_val = o[i][keepobs[i]].mean()
        for c in range(1, n + 1):
            cm = lab == c
            area = int(cm.sum())
            b = np.searchsorted(AREA_EDGES, area, side='right') - 1
            em = np.abs(p[i][cm] - o[i][cm]).sum()
            ef = np.abs(mf_val - o[i][cm]).sum()
            acc[b]['n'] += area
            acc[b]['model'] += em
            acc[b]['mf'] += ef
            acc[b]['holes'] += 1


@torch.no_grad()
def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    cfg = phase2(predict=args.predict)
    if args.smooth_sigma is not None:
        cfg.smooth_sigma = args.smooth_sigma
    print(f"device={dev} ({gpu})  ckpt={args.ckpt}  smooth_target={args.smooth_target}", flush=True)

    ds = RealDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='test',
                     fake_mask_frac=cfg.fake_mask_frac, fake_mask_mode=cfg.fake_mask_mode,
                     smooth_target=args.smooth_target, smooth_sigma=cfg.smooth_sigma)
    print(f"test baselines: {len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    print("model built, sampling", flush=True)

    cvis_m, cvis_f, ph_m, ph_f = [], [], [], []
    acc = [{'n': 0, 'model': 0.0, 'mf': 0.0, 'holes': 0} for _ in range(len(AREA_EDGES) - 1)]
    seen = 0
    t0 = time.time()
    for bi, batch in enumerate(dl):
        obs = batch['obs'].to(dev); hidden = batch['hidden'].to(dev); fake = batch['fake_mask'].to(dev)
        cond = build_cond(obs, hidden, batch['pe'].to(dev), hole_fill=getattr(cfg, 'hole_fill', 'mean'))
        pred = diff.sample(model, cond, obs, hidden, predict=cfg.predict, eta=0.0, steps=args.steps)
        base = meanfill(obs, hidden)

        cvis_m.append(float(complex_mae(pred, obs, fake)))
        cvis_f.append(float(complex_mae(base, obs, fake)))
        ph_m.append(float(phase_error(pred, obs, fake)))
        ph_f.append(float(phase_error(base, obs, fake)))
        stratify_amp(pred, obs, fake, hidden, acc)

        seen += obs.shape[0]
        rate = seen / (time.time() - t0)
        print(f"  batch {bi+1}  seen {seen}  {rate:.2f} bl/s  "
              f"cvis m={np.mean(cvis_m):.4f} mf={np.mean(cvis_f):.4f}", flush=True)
        if seen >= args.max_eval:
            break

    print(f"\nRESULT  {args.tag}  (n={seen})", flush=True)
    print(f"  complex-vis MAE   model {np.mean(cvis_m):.4f}   mean-fill {np.mean(cvis_f):.4f}   "
          f"{'MODEL WINS' if np.mean(cvis_m) < np.mean(cvis_f) else 'mean-fill wins'}", flush=True)
    print(f"  phase error (rad) model {np.mean(ph_m):.4f}   mean-fill {np.mean(ph_f):.4f}   "
          f"{'MODEL WINS' if np.mean(ph_m) < np.mean(ph_f) else 'mean-fill wins'}", flush=True)
    print("  amplitude MAE by hole size (model vs mean-fill):", flush=True)
    names = ['<64px', '64-256', '256-1024', '>1024']
    for b in range(len(acc)):
        a = acc[b]
        if a['n'] == 0:
            continue
        m = a['model'] / a['n']; f = a['mf'] / a['n']
        tag = 'MODEL WINS' if m < f else 'mean-fill'
        print(f"    {names[b]:>9}  holes={a['holes']:4d}  model {m:.4f}  mean-fill {f:.4f}  {tag}", flush=True)
    print(f"RESULTLINE\t{args.tag}\t{np.mean(cvis_m):.4f}\t{np.mean(cvis_f):.4f}\t"
          f"{np.mean(ph_m):.4f}\t{np.mean(ph_f):.4f}\t{seen}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-eval', type=int, default=64)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--smooth-target', action='store_true', dest='smooth_target')
    ap.add_argument('--smooth-sigma', type=float, default=None, dest='smooth_sigma')
    main(ap.parse_args())
