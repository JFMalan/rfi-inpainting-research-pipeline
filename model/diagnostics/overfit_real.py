import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase2
from data import RealDataset, fake_mask, build_cond, positional_encoding
from diffusion import Diffusion
from unet import UNet
from metrics import mae, tre


def synthetic_batch(n, sz, pe_channels, band_min=900.0, band_max=1650.0, seed=0):
    # in-memory stand-in for real baselines so the training contract can be tested
    # without an extracted dataset: structured fringe amplitude + a few real-like
    # flag bands. amplitude has recoverable structure (diagonal fringes) so a working
    # model MUST beat mean-fill — if it can't overfit this, the wiring is broken.
    rng = np.random.default_rng(seed)
    out = []
    pe = positional_encoding(band_min, band_max, band_min, band_max, sz, sz, pe_channels)
    for _ in range(n):
        x = np.linspace(0, 1, sz)
        tt, ff = np.meshgrid(x, x, indexing='ij')
        amp = (1.0 + 0.5 * np.sin(2 * np.pi * (3 * tt + 5 * ff) + rng.uniform(0, 6))
               + 0.3 * np.sin(2 * np.pi * (7 * ff) + rng.uniform(0, 6))).astype(np.float32)
        amp += rng.normal(0, 0.05, amp.shape).astype(np.float32)
        phase = (np.pi * np.sin(2 * np.pi * (2 * tt + 4 * ff))).astype(np.float32)
        rf = np.zeros((sz, sz), np.float32)
        for _ in range(4):
            w = rng.integers(10, 40); f0 = rng.integers(0, sz - w)
            rf[:, f0:f0 + w] = 1.0
        fm = fake_mask(rf)
        obs = np.stack([amp, np.cos(phase), np.sin(phase)], 0)
        hidden = np.clip(rf + fm, 0, 1)
        out.append({
            'obs': torch.from_numpy(obs),
            'real_flags': torch.from_numpy(rf)[None],
            'fake_mask': torch.from_numpy(fm)[None],
            'hidden': torch.from_numpy(hidden)[None],
            'pe': torch.from_numpy(pe.copy()),
        })
    return {k: torch.stack([o[k] for o in out]) for k in out[0]}


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    cfg = phase2(predict=args.predict)
    cfg.hole_fill = args.hole_fill
    torch.manual_seed(0)
    print(f"device={dev} ({gpu})  predict={cfg.predict}  in_channels={cfg.in_channels}  "
          f"{'SYNTHETIC' if args.synthetic else 'REAL'} data", flush=True)

    if args.synthetic:
        full = synthetic_batch(args.n, cfg.img_size, cfg.pe_channels)
    else:
        ds = RealDataset(args.data, pe_channels=cfg.pe_channels, augment=False,
                         split='train', max_patches=args.n)
        full = {k: torch.stack([ds[i][k] for i in range(min(args.n, len(ds)))]) for k in ds[0]}
    n = full['obs'].shape[0]
    bs = min(args.bs, n)
    print(f"overfitting {n} baselines on {dev}  (batch {bs}, {cfg.img_size}x{cfg.img_size})", flush=True)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size).to(dev)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(0)
    model.train()
    t0 = time.time()
    for it in range(args.iters):
        idx = torch.randint(0, n, (bs,), generator=g)
        mb = {k: full[k][idx].to(dev) for k in full}
        opt.zero_grad()
        loss = diff.loss_phase2(model, mb, cfg)
        loss.backward()
        opt.step()
        if it == 0 or (it + 1) % 50 == 0:
            rate = (it + 1) / max(time.time() - t0, 1e-6)
            print(f"  iter {it+1}/{args.iters}  loss {loss.item():.4f}  ({rate:.2f} it/s)", flush=True)

    model.eval()
    ne = min(n, args.eval_n)
    with torch.no_grad():
        obs = full['obs'][:ne].to(dev)
        hidden = full['hidden'][:ne].to(dev)
        fake = full['fake_mask'][:ne].to(dev)
        cond = build_cond(obs, hidden, full['pe'][:ne].to(dev), hole_fill=cfg.hole_fill)
        pred = diff.sample(model, cond, obs, hidden, predict=cfg.predict, eta=args.eta, steps=200)

        model_mae = float(mae(pred, obs, fake))
        tre_val = float(tre(pred, obs, fake))
        base = obs.clone()
        keep = hidden == 0
        for i in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                base[i, c] = obs[i, c][keep[i, 0]].mean()
        mf_mae = float(mae(base, obs, fake))
        mf_tre = float(tre(base, obs, fake))

    print(f"\n{'='*60}")
    print(f"PHASE-2 OVERFIT BENCHMARK  (fake-mask region, model vs mean-fill)")
    print(f"{'='*60}")
    print(f"  fake-mask amp MAE   model {model_mae:.4f}   mean-fill {mf_mae:.4f}")
    print(f"  TRE                 model {tre_val:.4f}   mean-fill {mf_tre:.4f}")
    print(f"{'='*60}")
    print("VERDICT:", "PASS (model beats mean-fill on fake holes -> learns to inpaint)"
          if model_mae < mf_mae - 0.005
          else "FAIL (model ~= mean-fill, not learning)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=None)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--eval-n', type=int, default=4, dest='eval_n')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--predict', default='x0', choices=['noise', 'x0'])
    ap.add_argument('--hole-fill', default='mean', choices=['zero', 'mean', 'noise', 'center'], dest='hole_fill')
    ap.add_argument('--eta', type=float, default=0.0)
    main(ap.parse_args())
