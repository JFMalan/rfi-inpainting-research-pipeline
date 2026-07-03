import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase2
from data import positional_encoding, build_cond
from diffusion import Diffusion
from unet import UNet

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def green(mask2d):
    rgba = np.zeros((*mask2d.T.shape, 4), np.float32)
    rgba[mask2d.T > 0] = [0.0, 1.0, 0.2, 0.85]
    return rgba


def load_model(ckpt, cfg, dev):
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    return model


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    labels = [c.split('=', 1)[0] for c in args.ckpts]
    paths = [c.split('=', 1)[1] for c in args.ckpts]

    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    band_min = float(hf.attrs['freq_min_mhz'])
    band_max = float(hf.attrs['freq_max_mhz'])
    n = hf['data'].shape[0]
    split = hf['split'][:] if 'split' in hf else np.ones(n, np.int32)
    has_fpatch = 'freq_min_patch' in hf
    flags_frac = hf['flags'][:].astype(np.float32).mean(axis=(1, 2))

    test = np.where((split == 1) & (flags_frac >= args.min_flag_frac)
                    & (flags_frac < args.max_flag_frac))[0]
    rng = np.random.default_rng(args.seed)
    if len(test) > args.n_show:
        test = np.sort(rng.choice(test, args.n_show, replace=False))
    log(f"device={dev} ({gpu})  h5={args.h5}  tiles={len(test)} "
        f"(split==1, {args.min_flag_frac}<=flagfrac<{args.max_flag_frac})  nf={args.noise_floor}")

    cfg = phase2(predict=args.predict)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    models = []
    for lb, p in zip(labels, paths):
        models.append(load_model(p, cfg, dev))
        log(f"loaded {lb} <- {p}")
    nf = None if args.noise_floor in ('none', 'None') else float(args.noise_floor)

    rows = []
    for k, u in enumerate(test):
        data = hf['data'][u].astype(np.float32)
        phase = hf['phase'][u].astype(np.float32)
        flags = hf['flags'][u].astype(np.float32)
        if has_fpatch:
            fmin = float(hf['freq_min_patch'][u]); fmax = float(hf['freq_max_patch'][u])
        else:
            fmin, fmax = band_min, band_max
        pe = positional_encoding(fmin, fmax, band_min, band_max, sz, sz, cfg.pe_channels)
        x0 = torch.from_numpy(np.stack([data, np.cos(phase), np.sin(phase)], 0)[None]).to(dev)
        m = torch.from_numpy(flags[None, None]).to(dev)
        pe_b = torch.from_numpy(pe[None].copy()).to(dev)
        cond = build_cond(x0, m, pe_b, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
        fills = []
        with torch.no_grad():
            for model in models:
                pr = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0,
                                 steps=args.steps, noise_floor=nf).cpu().numpy()[0]
                fills.append(np.where(flags > 0.5, pr[0], data))
        rows.append((int(u), data, flags, fmin, fmax, fills))
        log(f"  tile {k + 1}/{len(test)} (unit {int(u)}, flag={flags.mean():.2f})")
    hf.close()

    ncols = 2 + len(models)
    n = len(rows)
    fig, axes = plt.subplots(n, ncols, figsize=(3.7 * ncols, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    titles = ["observed amp (RFI)", "RFI mask"] + [f"{lb} fill" for lb in labels]
    for r, (u, data, flags, fmin, fmax, fills) in enumerate(rows):
        trust = flags == 0
        src = data[trust] if trust.any() else data
        vmin, vmax = np.percentile(src, 1), np.percentile(src, 99)
        ext = [0, data.shape[0], fmin, fmax]
        axes[r, 0].imshow(data.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 1].imshow(data.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 1].imshow(green(flags), aspect='auto', origin='lower', extent=ext)
        for j, fl in enumerate(fills):
            axes[r, 2 + j].imshow(fl.T, aspect='auto', origin='lower', extent=ext,
                                  vmin=vmin, vmax=vmax, cmap='plasma')
        axes[r, 0].set_ylabel(f"unit {u}\nFreq (MHz)\nflag={flags.mean():.2f}\n"
                              f"scale[{vmin:.2f},{vmax:.2f}]", fontsize=7)
        if r == 0:
            for j, t in enumerate(titles):
                axes[r, j].set_title(t, fontsize=9)
        for ax in axes[r]:
            ax.tick_params(labelsize=6); ax.set_xlabel("Time bin", fontsize=7)

    plt.tight_layout()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"-> {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--ckpts', nargs='+', required=True,
                    help="label=path entries, e.g. sim=/.../best.pt finetune=/.../best.pt scratch=/.../best.pt")
    ap.add_argument('--output', required=True)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--noise-floor', default='0.5', dest='noise_floor')
    ap.add_argument('--n-show', type=int, default=20, dest='n_show')
    ap.add_argument('--min-flag-frac', type=float, default=0.15, dest='min_flag_frac')
    ap.add_argument('--max-flag-frac', type=float, default=0.6, dest='max_flag_frac')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
