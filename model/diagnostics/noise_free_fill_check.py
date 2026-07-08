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

from config import phase1
from data import positional_encoding, build_cond
from diffusion import Diffusion
from unet import UNet

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def delay_power(amp):
    w = np.blackman(amp.shape[1])[None, :]
    ft = np.fft.rfft(amp * w, axis=1)
    return (np.abs(ft) ** 2).mean(axis=0)


def smooth_freq(a, sigma=2.0):
    rad = int(3 * sigma)
    x = np.arange(-rad, rad + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode='same'), 1, a)


def local_sigma(amp, known):
    r = (amp - smooth_freq(amp))[known]
    return 1.4826 * np.median(np.abs(r - np.median(r))) if r.size else 0.0


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    cfg = phase1(predict='x0')
    diff = Diffusion(T=cfg.timesteps, device=dev)

    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    band_min = float(hf.attrs['freq_min_mhz']); band_max = float(hf.attrs['freq_max_mhz'])
    flagfrac = hf['mask'][:].astype(np.float32).mean(axis=(1, 2))
    cand = np.where((flagfrac >= args.min_flag) & (flagfrac < args.max_flag))[0]
    rng = np.random.default_rng(args.seed)
    tiles = np.sort(rng.choice(cand, min(args.n_show, len(cand)), replace=False))
    log(f"device={dev} ({gpu})  h5={args.h5}  tiles={list(tiles)}  nf sweep=[none, matched local sigma]")

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    log(f"loaded {args.ckpt}")

    rows = []
    for k, u in enumerate(tiles):
        clean = hf['clean'][u].astype(np.float32)          # noise-free amp target
        corr = hf['corrupted'][u].astype(np.float32)       # noisy + RFI (context)
        mask = hf['mask'][u].astype(np.float32)
        phase = hf['phase'][u].astype(np.float32)
        fmin = float(hf['freq_min_patch'][u]); fmax = float(hf['freq_max_patch'][u])
        pe = positional_encoding(fmin, fmax, band_min, band_max, sz, sz, cfg.pe_channels)
        obs = np.stack([corr, np.cos(phase), np.sin(phase)], 0)[None]
        x0 = torch.from_numpy(obs).to(dev)
        m = torch.from_numpy(mask[None, None]).to(dev)
        pe_b = torch.from_numpy(pe[None].copy()).to(dev)
        cond = build_cond(x0, m, pe_b, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
        with torch.no_grad():
            pred = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0,
                               steps=args.steps, noise_floor=None).cpu().numpy()[0]
        mk = mask > 0.5
        smooth = np.where(mk, pred[0], corr)
        sig = local_sigma(corr, ~mk)
        grain = smooth + mk * rng.normal(0.0, sig, corr.shape).astype(np.float32)
        rows.append((int(u), corr, clean, smooth, grain, mk, sig))
        dp = {n: delay_power(a) for n, a in [('obs', corr), ('truth', clean), ('smooth', smooth), ('grain', grain)]}
        hi = slice(len(dp['obs']) // 2, None)
        log(f"  tile {u}: flag={mask.mean():.2f} sigma_local={sig:.3f}  "
            f"hi-delay P grain/obs={dp['grain'][hi].mean()/dp['obs'][hi].mean():.2f} "
            f"smooth/obs={dp['smooth'][hi].mean()/dp['obs'][hi].mean():.2f}")
    hf.close()

    ncol = 5
    fig, ax = plt.subplots(len(rows), ncol, figsize=(3.2 * ncol, 3.0 * len(rows)))
    ax = np.atleast_2d(ax)
    titles = ['observed (noisy+RFI)', 'target (noise-free)', 'fill: smooth (nf=none)',
              'fill: +matched grain', 'delay spectrum']
    for r, (u, corr, clean, smooth, grain, mk, sig) in enumerate(rows):
        vmin, vmax = np.percentile(corr[~mk], 1), np.percentile(corr[~mk], 99)
        for j, img in enumerate([corr, clean, smooth, grain]):
            ax[r, j].imshow(img.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='plasma')
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if r == 0:
                ax[r, j].set_title(titles[j], fontsize=9)
        for n, a, c in [('obs', corr, 'k'), ('truth', clean, 'C2'),
                        ('smooth', smooth, 'C1'), ('grain', grain, 'C0')]:
            ax[r, 4].semilogy(delay_power(a), c, lw=1, label=n)
        ax[r, 4].set_xlabel('delay bin', fontsize=7); ax[r, 4].tick_params(labelsize=6)
        if r == 0:
            ax[r, 4].set_title(titles[4], fontsize=9); ax[r, 4].legend(fontsize=6)
        ax[r, 0].set_ylabel(f"tile {u}\nsigma_loc={sig:.3f}", fontsize=8)

    plt.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    log(f"-> {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True, help='paired dataset (noisy context, noise-free target)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--output', required=True, dest='out')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--n-show', type=int, default=4, dest='n_show')
    ap.add_argument('--min-flag', type=float, default=0.1, dest='min_flag')
    ap.add_argument('--max-flag', type=float, default=0.5, dest='max_flag')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
