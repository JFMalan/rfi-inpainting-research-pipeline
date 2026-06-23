import argparse
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase1
from data import positional_encoding, build_cond, smooth_component
from diffusion import Diffusion
from unet import UNet

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def load_patches(path, n, rng, smooth_sigma=1.0):
    with h5py.File(path, 'r') as f:
        tot = f['clean'].shape[0]
        idx = np.sort(rng.choice(tot, min(n, tot), replace=False))
        clean = f['clean'][idx].astype(np.float32)
        mask = f['mask'][idx].astype(np.float32)
        phase = f['phase'][idx].astype(np.float32)
        nt = int(f.attrs['n_time']); nf = int(f.attrs['n_freq'])
        bmin = float(f.attrs['freq_min_mhz']); bmax = float(f.attrs['freq_max_mhz'])

    smooth = np.stack([
        smooth_component(clean[i], mask[i], sigma=smooth_sigma)
        for i in range(len(idx))
    ], axis=0)

    pe = positional_encoding(bmin, bmax, bmin, bmax, nf, nt, 4)
    cos, sin = np.cos(phase), np.sin(phase)
    clean3 = np.stack([clean, cos, sin], 1)
    smooth3 = np.stack([smooth, cos, sin], 1)
    return (torch.from_numpy(clean3), torch.from_numpy(smooth3),
            torch.from_numpy(mask)[:, None],
            torch.from_numpy(pe[None]).repeat(len(idx), 1, 1, 1))


def texture_ratio(pred, target, mask):
    # std of 5x5 high-pass residual in hole (pred) vs known region (target).
    # ratio 1.0 = matched noise floor, 0.0 = smooth fill.
    B, C, H, W = pred.shape
    def hp(x):
        pad = F.pad(x, (2, 2, 2, 2), mode='reflect')
        return x - F.avg_pool2d(pad, 5, stride=1)
    hole = mask[:, 0] > 0          # (B, H, W)
    known = ~hole
    ratios = []
    for b in range(B):
        hp_p = hp(pred[b:b+1]).squeeze(0)   # (C, H, W)
        hp_t = hp(target[b:b+1]).squeeze(0)
        p_std = hp_p[:, hole[b]].std().item()
        t_std = hp_t[:, known[b]].std().item()
        if t_std > 1e-6:
            ratios.append(p_std / t_std)
    return float(np.mean(ratios)) if ratios else 0.0


def mae_in_hole(pred, target, mask):
    hole = mask > 0
    return (pred - target).abs()[hole].mean().item()


def run_condition(diff, model, clean, mask, pe, cfg, eta, noise_floor, steps, dev, label):
    log(f"  running: {label}")
    preds = []
    bs = 4
    for s in range(0, clean.shape[0], bs):
        e = min(s + bs, clean.shape[0])
        cb = clean[s:e].to(dev); mb = mask[s:e].to(dev); pb = pe[s:e].to(dev)
        cond = build_cond(cb, mb, pb, hole_fill='mean')
        p = diff.sample(model, cond, cb, mb, predict=cfg.predict, eta=eta,
                        steps=steps, noise_floor=noise_floor)
        preds.append(p.cpu())
        if (s // bs + 1) % 4 == 0:
            log(f"    {e}/{clean.shape[0]} patches  ({time.time()-t0:.0f}s)")
    return torch.cat(preds, 0)


def meanfill(clean, mask):
    out = clean.clone()
    for i in range(clean.shape[0]):
        km = mask[i, 0] == 0
        for c in range(clean.shape[1]):
            out[i, c][mask[i, 0] > 0] = clean[i, c][km].mean()
    return out


def interp_fill(clean, mask):
    out = clean.clone()
    a = clean[:, 0].numpy(); h = (mask[:, 0] > 0).numpy()
    nf = a.shape[-1]; idx = np.arange(nf)
    for i in range(a.shape[0]):
        for tt in range(a.shape[1]):
            hr = h[i, tt]
            if hr.any() and not hr.all():
                a[i, tt, hr] = np.interp(idx[hr], idx[~hr], a[i, tt, ~hr])
    out[:, 0] = torch.from_numpy(a)
    return out


def save_grid(clean, smooth, preds_dict, mask, out_path, n_show=6):
    n = min(n_show, clean.shape[0])
    # pick patches with non-trivial holes
    scores = [(mask[i, 0].sum().item(), i) for i in range(clean.shape[0])]
    idxs = [i for _, i in sorted(scores, reverse=True)[:n]]

    labels = ['clean (noisy)'] + list(preds_dict.keys())
    sources = [clean[:, 0]] + [v[:, 0] for v in preds_dict.values()]

    fig, axes = plt.subplots(len(labels), n, figsize=(n * 3, len(labels) * 2.5))
    if n == 1:
        axes = axes[:, None]

    for col, i in enumerate(idxs):
        for row, (label, src) in enumerate(zip(labels, sources)):
            ax = axes[row, col]
            img = src[i].numpy()
            vmin, vmax = clean[i, 0].numpy().min(), clean[i, 0].numpy().max()
            ax.imshow(img, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='inferno')
            if col == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('Stochastic inpainting conditions — amplitude channel', fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    log(f"grid saved -> {out_path}")


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"device={dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})  "
        f"ckpt={Path(args.ckpt).name}  data={Path(args.data).name}  n={args.n}")

    rng = np.random.default_rng(args.seed)
    clean, smooth, mask, pe = load_patches(args.data, args.n, rng, smooth_sigma=args.smooth_sigma)
    log(f"loaded {clean.shape[0]} patches  clean_std={clean[:,0].std():.3f}  "
        f"smooth_std={smooth[:,0].std():.3f}  hole_frac={mask.mean():.3f}")

    cfg = phase1(predict=args.predict)
    model = UNet(cfg.in_channels, out_ch=3, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    log("model loaded")

    conditions = [
        ('eta=0  no_noise',   0.0,  None),
        ('eta=0  +auto_noise', 0.0,  'auto'),
        ('eta=1  no_noise',   1.0,  None),
        ('eta=1  +auto_noise', 1.0,  'auto'),
    ]

    mf = meanfill(clean, mask)
    ip = interp_fill(clean, mask)

    results = {}
    results['mean_fill'] = mf
    results['interp_freq'] = ip

    with torch.no_grad():
        for label, eta, nf in conditions:
            results[label] = run_condition(diff, model, clean, mask, pe, cfg,
                                           eta, nf, args.steps, dev, label)

    log("all conditions done, computing metrics")

    header = f"{'CONDITION':<25}  {'TEXTURE':>7}  {'MAE_noisy':>9}  {'MAE_smooth':>10}"
    sep = '-' * len(header)
    print(f"\n{sep}", flush=True)
    print(f"STOCHASTIC INPAINT PROBE  n={clean.shape[0]}  steps={args.steps}  "
          f"predict={args.predict}", flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    for name, pred in results.items():
        tr = texture_ratio(pred, clean, mask)
        mn = mae_in_hole(pred[:, 0:1], clean[:, 0:1], mask)
        ms = mae_in_hole(pred[:, 0:1], smooth[:, 0:1], mask)
        print(f"{name:<25}  {tr:>7.3f}  {mn:>9.4f}  {ms:>10.4f}", flush=True)

    print(sep, flush=True)
    print("texture_ratio: 1.0 = matched noise floor, 0.0 = smooth fill", flush=True)
    print("MAE_smooth: fair comparison (recoverable signal only)", flush=True)
    print(sep, flush=True)

    if args.out_png:
        save_grid(clean, smooth, results, mask, args.out_png)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smooth-sigma', type=float, default=1.0, dest='smooth_sigma')
    ap.add_argument('--out-png', default='', dest='out_png')
    main(ap.parse_args())
