import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase2
from data import positional_encoding, build_cond
from diffusion import Diffusion
from unet import UNet


def lowpass(amp, method, sigma):
    # smooth (low-frequency) component. high = amp - low is exact, so low+high==amp
    # everywhere -> the decomposition is mathematically reversible by construction.
    if method == 'gaussian':
        return gaussian_filter(amp, sigma=sigma, mode='nearest')
    if method == 'median':
        k = max(3, int(round(sigma * 2)) | 1)
        return median_filter(amp, size=k, mode='nearest')
    if method == 'wavelet':
        # 2-level Laplacian-pyramid low band via repeated Gaussian (still low+high=amp exact)
        return gaussian_filter(gaussian_filter(amp, sigma=sigma, mode='nearest'),
                               sigma=sigma, mode='nearest')
    raise ValueError(method)


def decompose(data, flags, method, sigma):
    # hole-fill (interp over freq) ONLY to compute a sane low-pass; high is taken as the
    # true residual on observed pixels so reconstruction outside the hole is exact.
    filled = data.copy()
    nf = data.shape[1]; idx = np.arange(nf)
    for t in range(data.shape[0]):
        keep = flags[t] < 0.5
        if keep.sum() >= 4:
            filled[t] = np.interp(idx, idx[keep], data[t][keep])
        else:
            filled[t] = data[t].mean() if keep.any() else 1.0
    low = lowpass(filled, method, sigma).astype(np.float32)
    high = (data - low).astype(np.float32)          # exact residual on observed pixels
    return low, high


def resample_high(high, flags, fake_or_real_hole, rng):
    # inside the hole the true high-freq detail is unrecoverable -> draw white noise at the
    # locally-measured residual std (matches the surrounding grain). outside the hole keep
    # the TRUE high (exact reverse).
    hole = fake_or_real_hole > 0.5
    obs = flags < 0.5
    std = high[obs].std() if obs.any() else 0.0
    out = high.copy()
    if std > 0:
        out[hole] = rng.standard_normal(int(hole.sum())).astype(np.float32) * std
    return out


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase2(predict=args.predict)
    print(f"device={dev}  ckpt={args.ckpt}  methods={args.methods}", flush=True)

    f = h5py.File(args.data, 'r')
    n_t = int(f.attrs['n_time']); n_f = int(f.attrs['n_freq'])
    bmin = float(f.attrs['freq_min_mhz']); bmax = float(f.attrs['freq_max_mhz'])
    ntot = f['data'].shape[0]
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(ntot, size=min(args.n, ntot), replace=False).tolist())

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    pe = positional_encoding(bmin, bmax, bmin, bmax, n_f, n_t, cfg.pe_channels)
    methods = args.methods.split(',')

    rng_h = np.random.default_rng(0)
    rows = []
    with torch.no_grad():
        for k, i in enumerate(idxs):
            data = f['data'][i].astype(np.float32)
            phase = f['phase'][i].astype(np.float32)
            flags = f['flags'][i].astype(np.float32)
            per_method = {}
            recon_err = {}
            for mth in methods:
                low, high = decompose(data, flags, mth, args.sigma)
                # reversibility check on observed pixels: low+high must equal data exactly
                obs = flags < 0.5
                recon_err[mth] = float(np.abs((low + high - data)[obs]).max())
                # inpaint the LOW component with the model (predicts smooth structure)
                obs_stack = np.stack([low, np.cos(phase), np.sin(phase)], 0)
                x0 = torch.from_numpy(obs_stack)[None].to(dev)
                m = torch.from_numpy(flags)[None, None].to(dev)
                pe_t = torch.from_numpy(pe.copy())[None].to(dev)
                cond = build_cond(x0, m, pe_t, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
                low_filled = diff.sample(model, cond, x0, m, predict=cfg.predict,
                                         eta=0.0, steps=args.steps)[0, 0].cpu().numpy()
                # reverse: inside the RFI band the data is fully corrupt -> there is NO
                # reliable high to invert, so synthesise it (low_filled + resampled noise).
                # outside the band we keep the UNTOUCHED observation exactly (true reverse).
                high_r = resample_high(high, flags, flags, rng_h)
                hole = flags > 0.5
                result = np.where(hole, low_filled + high_r, data).astype(np.float32)
                per_method[mth] = (low, low_filled, result)
            rows.append((data, flags, per_method, recon_err, i))
            errstr = "  ".join(f"{m}:{recon_err[m]:.1e}" for m in methods)
            print(f"  sampled {k+1}/{len(idxs)}  recon_err {errstr}", flush=True)
    f.close()

    # one row per baseline; columns: observed | mask | [per method: low, low_filled, result]
    nm = len(methods)
    ncol = 2 + 3 * nm
    n = len(rows)
    fig, ax = plt.subplots(n, ncol, figsize=(3.3 * ncol, 3.2 * n))
    if n == 1:
        ax = ax[None, :]
    for r, (data, flags, per_method, recon_err, idx) in enumerate(rows):
        obs = flags < 0.5
        vmin = np.percentile(data[obs], 1); vmax = np.percentile(data[obs], 99)
        ext = [0, n_t, bmin, bmax]
        def show(col, img, title):
            ax[r, col].imshow(img.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
            if r == 0: ax[r, col].set_title(title, fontsize=8)
            ax[r, col].tick_params(labelsize=5)
        show(0, data, 'observed (RFI)')
        ax[r, 1].imshow(data.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
        g = np.zeros((*flags.T.shape, 4), np.float32); g[flags.T > 0.5] = [0, 1, 0.2, 0.8]
        ax[r, 1].imshow(g, aspect='auto', origin='lower', extent=ext)
        if r == 0: ax[r, 1].set_title('RFI mask', fontsize=8)
        c = 2
        for mth in methods:
            low, low_filled, result = per_method[mth]
            show(c, low, f'{mth}: low'); show(c+1, low_filled, f'{mth}: low inpainted')
            show(c+2, result, f'{mth}: REVERSED (+noise)'); c += 3
        ax[r, 0].set_ylabel(f"bl {idx} flag={flags.mean():.2f}\n"
                            f"recon_err={max(recon_err.values()):.1e}", fontsize=6)
    plt.tight_layout()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close()
    maxerr = max(max(re.values()) for _, _, _, re, _ in rows)
    print(f"-> {out}", flush=True)
    print(f"max reconstruction error on observed pixels (should be ~0): {maxerr:.2e}", flush=True)
    print("  (confirms low+high==data exactly outside the hole -> reverse is lossless there)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--methods', default='gaussian,median,wavelet')
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
