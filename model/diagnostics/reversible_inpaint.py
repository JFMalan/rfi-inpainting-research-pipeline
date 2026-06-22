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


def _nc_gauss(data, valid, sigma):
    # normalized convolution: Gaussian-smooth using ONLY unflagged pixels, so flagged
    # bands are filled by distance-weighted REAL neighbours (no spurious interp values
    # smeared in). num/den both blurred -> low in a band reflects nearby real data only.
    num = gaussian_filter(data * valid, sigma=sigma, mode='nearest')
    den = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode='nearest')
    return num / np.clip(den, 1e-3, None)


def lowpass(data, valid, method, sigma):
    # hole-AWARE low-pass: built only from unflagged pixels (kills the interp-across-band
    # vertical-streak artifact). high = data - low is still exact on observed pixels.
    if method == 'gaussian':
        return _nc_gauss(data, valid, sigma)
    if method == 'median':
        # median over a hole-filled-by-NC base, then one NC pass to suppress streaks
        base = _nc_gauss(data, valid, sigma)
        k = max(3, int(round(sigma * 2)) | 1)
        return median_filter(base, size=k, mode='nearest')
    if method == 'wavelet':
        return _nc_gauss(_nc_gauss(data, valid, sigma), valid, sigma)
    raise ValueError(method)


def decompose(data, flags, method, sigma):
    valid = (flags < 0.5).astype(np.float32)
    low = lowpass(data, valid, method, sigma).astype(np.float32)
    high = (data - low).astype(np.float32)          # exact residual on observed pixels
    return low, high


def level_match(low_filled, low_ref, flags):
    # the model's in-band fill can sit at the wrong baseline level (the documented wide-band
    # bias) -> visible colour step at the edge. low_ref is the hole-aware NC low-pass, whose
    # in-band value IS the distance-weighted real-neighbour level. Offset each filled
    # frequency-channel (per time) so its in-hole mean matches low_ref's -> continuous edge.
    out = low_filled.copy()
    hole = flags > 0.5
    nt = low_filled.shape[0]
    for t in range(nt):
        h = hole[t]
        if h.any():
            off = low_ref[t][h].mean() - low_filled[t][h].mean()
            out[t][h] = low_filled[t][h] + off
    return out.astype(np.float32)


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
            obs = flags < 0.5
            hole = flags > 0.5
            # FORMAT PARITY: feed the model the REAL amplitude as known-region + conditioning,
            # exactly like sim training feeds clean amp (~1.0 centered). The model fills the hole
            # at the correct level+structure in its trained scale. (Earlier bug: feeding the
            # NC-'low' as context — dark in wide bands — collapsed the fill to a solid dark slab,
            # and level_match to that dark low made it worse. Inpaint on data, texture separately.)
            obs_stack = np.stack([data, np.cos(phase), np.sin(phase)], 0)
            x0 = torch.from_numpy(obs_stack)[None].to(dev)
            m = torch.from_numpy(flags)[None, None].to(dev)
            pe_t = torch.from_numpy(pe.copy())[None].to(dev)
            cond = build_cond(x0, m, pe_t, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
            model_fill = diff.sample(model, cond, x0, m, predict=cfg.predict,
                                     eta=0.0, steps=args.steps)[0, 0].cpu().numpy()
            for mth in methods:
                # decomposition is used ONLY to get the high-freq TEXTURE to add back;
                # the model fill already provides the correct level+structure.
                low, high = decompose(data, flags, mth, args.sigma)
                recon_err[mth] = float(np.abs((low + high - data)[obs]).max())
                # inside band: model fill (correct level) + resampled texture; outside: data.
                high_r = resample_high(high, flags, flags, rng_h)
                result = np.where(hole, model_fill + high_r, data).astype(np.float32)
                per_method[mth] = (low, model_fill, result)
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
