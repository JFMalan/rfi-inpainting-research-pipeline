import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase1
from data import positional_encoding, build_cond
from diffusion import Diffusion
from unet import UNet

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def load(path, n, rng):
    with h5py.File(path, 'r') as f:
        tot = f['clean'].shape[0]
        idx = np.sort(rng.choice(tot, min(n, tot), replace=False))
        clean = f['clean'][idx].astype(np.float32)
        smooth = f['clean_smooth'][idx].astype(np.float32) if 'clean_smooth' in f else clean
        mask = f['mask'][idx].astype(np.float32)
        phase = f['phase'][idx].astype(np.float32)
        nt = int(f.attrs['n_time']); nf = int(f.attrs['n_freq'])
        bmin = float(f.attrs['freq_min_mhz']); bmax = float(f.attrs['freq_max_mhz'])
    pe = positional_encoding(bmin, bmax, bmin, bmax, nf, nt, 4)
    cos, sin = np.cos(phase), np.sin(phase)
    clean3 = np.stack([clean, cos, sin], 1)
    smooth3 = np.stack([smooth, cos, sin], 1)
    return (torch.from_numpy(clean3), torch.from_numpy(smooth3),
            torch.from_numpy(mask)[:, None], torch.from_numpy(pe[None]).repeat(len(idx), 1, 1, 1))


def baselines(clean, smooth, mask):
    # mean-fill and freq-interp baselines on amplitude, scored vs BOTH targets
    amp = clean[:, 0:1]; sm = smooth[:, 0:1]
    region = mask > 0
    out = {}
    mf = torch.zeros_like(amp)
    interp = amp.clone()
    for i in range(amp.shape[0]):
        kn = amp[i][mask[i] == 0]
        mf[i] = kn.mean()
        a = amp[i, 0].numpy(); h = (mask[i, 0] > 0).numpy()
        idx = np.arange(a.shape[1])
        for tt in range(a.shape[0]):
            hr = h[tt]
            if hr.any() and not hr.all():
                a[tt, hr] = np.interp(idx[hr], idx[~hr], a[tt, ~hr])
        interp[i, 0] = torch.from_numpy(a)
    for tgt, name in [(amp, 'vs_noisy'), (sm, 'vs_smooth')]:
        out[f'meanfill_{name}'] = (mf - tgt).abs()[region].mean().item()
        out[f'interp_{name}'] = (interp - tgt).abs()[region].mean().item()
    return out


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(0)
    cfg = phase1(predict=args.predict)
    cfg.hole_fill = 'mean'
    torch.manual_seed(0)
    log(f"device={dev} ({torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'})  "
        f"predict={args.predict} eta={args.eta}  data={Path(args.data).name}")

    clean, smooth, mask, pe = load(args.data, args.n, rng)
    log(f"loaded {clean.shape[0]} patches  clean_std={clean[:,0].std():.3f}  "
        f"smooth_std={smooth[:,0].std():.3f}")
    # keep the full set on CPU; only the active mini-batch goes to GPU (matches train.py)

    model = UNet(cfg.in_channels, out_ch=3, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0)

    # decompose-then-inpaint: target = recoverable smooth structure, context = noisy obs.
    # 'noisy' reproduces the old behaviour (target = noisy clean).
    tgt_all = smooth if args.train_target == 'smooth' else clean
    log(f"train target = {args.train_target} (std {tgt_all[:,0].std():.3f}); "
        f"context = noisy clean (std {clean[:,0].std():.3f})")

    model.train()
    win = []
    for it in range(args.iters):
        idx = torch.randint(0, clean.shape[0], (args.bs,), generator=g)
        mb = {'clean': tgt_all[idx].to(dev), 'corrupted': clean[idx].to(dev),
              'mask': mask[idx].to(dev), 'pe': pe[idx].to(dev)}
        opt.zero_grad()
        loss = diff.loss(model, mb, cfg)
        loss.backward()
        opt.step()
        win.append(loss.item())
        if (it + 1) % args.log_every == 0:
            log(f"  iter {it+1}/{args.iters}  loss {np.mean(win[-args.log_every:]):.4f}")

    model.eval()
    with torch.no_grad():
        preds = []
        for s in range(0, clean.shape[0], args.eval_bs):
            e = min(s + args.eval_bs, clean.shape[0])
            cb, mb_, pb = clean[s:e].to(dev), mask[s:e].to(dev), pe[s:e].to(dev)
            cond = build_cond(cb, mb_, pb, hole_fill='mean')
            preds.append(diff.sample(model, cond, cb, mb_,
                                     predict=cfg.predict, eta=args.eta, steps=args.steps).cpu())
            log(f"  sampled {e}/{clean.shape[0]}")
        pred = torch.cat(preds, 0)
        clean, smooth, mask = clean.cpu(), smooth.cpu(), mask.cpu()
        region = mask > 0
        amp_p = pred[:, 0:1]
        mae_noisy = (amp_p - clean[:, 0:1]).abs()[region].mean().item()
        mae_smooth = (amp_p - smooth[:, 0:1]).abs()[region].mean().item()
        bl = baselines(clean.cpu(), smooth.cpu(), mask.cpu())

        ap = pred[:, 0].cpu().numpy(); at = clean[:, 0].cpu().numpy()
        mk = mask[:, 0].cpu().numpy() > 0
        trs = []
        for b in range(ap.shape[0]):
            if mk[b].sum() < 20 or (~mk[b]).sum() < 20:
                continue
            hp_p = ap[b] - uniform_filter(ap[b], 5, mode='nearest')
            hp_k = at[b] - uniform_filter(at[b], 5, mode='nearest')
            if hp_k[~mk[b]].std() > 1e-6:
                trs.append(hp_p[mk[b]].std() / hp_k[~mk[b]].std())
        texture = float(np.mean(trs)) if trs else 0.0

    print(f"\n{'='*64}", flush=True)
    print(f"SPECKLE PROBE  data={Path(args.data).name}  predict={args.predict} eta={args.eta}", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"amplitude MAE in mask:", flush=True)
    print(f"  vs NOISY target (clean+speckle):  model {mae_noisy:.4f}   "
          f"meanfill {bl['meanfill_vs_noisy']:.4f}   interp {bl['interp_vs_noisy']:.4f}", flush=True)
    print(f"  vs SMOOTH target (recoverable):   model {mae_smooth:.4f}   "
          f"meanfill {bl['meanfill_vs_smooth']:.4f}   interp {bl['interp_vs_smooth']:.4f}", flush=True)
    print(f"texture ratio: {texture:.3f}", flush=True)
    recov = mae_smooth < bl['meanfill_vs_smooth'] - 0.005 and mae_smooth < bl['interp_vs_smooth'] - 0.005
    print(f"\nVERDICT: {'RECOVERS smooth structure (beats baselines on recoverable target)' if recov else 'does NOT beat baselines even on smooth target'}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--eval-bs', type=int, default=8, dest='eval_bs')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--predict', default='x0', choices=['x0', 'noise'])
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--train-target', default='noisy', choices=['noisy', 'smooth'], dest='train_target')
    main(ap.parse_args())
