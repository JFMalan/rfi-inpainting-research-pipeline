import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase2
from data import positional_encoding, build_cond
from diffusion import Diffusion
from unet import UNet


def load_model(ckpt, cfg, dev):
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    return model


def green(mask2d):
    rgba = np.zeros((*mask2d.T.shape, 4), np.float32)
    rgba[mask2d.T > 0.5] = [0.0, 1.0, 0.2, 0.85]
    return rgba


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase2(predict=args.predict)
    print(f"device={dev}  sim={args.sim_ckpt}  finetune={args.ft_ckpt}", flush=True)

    f = h5py.File(args.data, 'r')
    n_t = int(f.attrs['n_time']); n_f = int(f.attrs['n_freq'])
    band_min = float(f.attrs['freq_min_mhz']); band_max = float(f.attrs['freq_max_mhz'])
    ntot = f['data'].shape[0]
    rng = np.random.default_rng(args.seed)
    idxs = sorted(rng.choice(ntot, size=min(args.n, ntot), replace=False).tolist())

    sim_model = load_model(args.sim_ckpt, cfg, dev)
    ft_model = load_model(args.ft_ckpt, cfg, dev)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    pe = positional_encoding(band_min, band_max, band_min, band_max, n_f, n_t, cfg.pe_channels)

    rows = []
    with torch.no_grad():
        for k, i in enumerate(idxs):
            data = f['data'][i].astype(np.float32)
            phase = f['phase'][i].astype(np.float32)
            flags = f['flags'][i].astype(np.float32)
            obs = np.stack([data, np.cos(phase), np.sin(phase)], 0)
            x0 = torch.from_numpy(obs)[None].to(dev)
            m = torch.from_numpy(flags)[None, None].to(dev)
            pe_t = torch.from_numpy(pe.copy())[None].to(dev)
            cond = build_cond(x0, m, pe_t, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
            sim_pred = diff.sample(sim_model, cond, x0, m, predict=cfg.predict, eta=0.0, steps=args.steps)[0].cpu().numpy()
            ft_pred = diff.sample(ft_model, cond, x0, m, predict=cfg.predict, eta=0.0, steps=args.steps)[0].cpu().numpy()
            rows.append((data, phase, flags, sim_pred, ft_pred, i))
            print(f"  sampled {k+1}/{len(idxs)}", flush=True)
    f.close()

    n = len(rows)
    titles = ["observed amp (RFI)", "RFI mask", "SIM-model fill", "FINETUNE fill",
              "observed phase", "SIM phase fill", "FINETUNE phase fill"]
    fig, ax = plt.subplots(n, 7, figsize=(4 * 7, 3.4 * n))
    if n == 1:
        ax = ax[None, :]
    for r, (dt, ph, fl, sp, fp, idx) in enumerate(rows):
        unflag = fl < 0.5
        vmin = np.percentile(dt[unflag], 1) if unflag.any() else float(dt.min())
        vmax = np.percentile(dt[unflag], 99) if unflag.any() else float(dt.max())
        ext = [0, n_t, band_min, band_max]
        sim_amp = np.where(fl > 0.5, sp[0], dt)
        ft_amp = np.where(fl > 0.5, fp[0], dt)
        sim_ph = np.where(fl > 0.5, np.arctan2(sp[2], sp[1]), ph)
        ft_ph = np.where(fl > 0.5, np.arctan2(fp[2], fp[1]), ph)
        ims = [(dt, 'plasma', vmin, vmax), None, (sim_amp, 'plasma', vmin, vmax),
               (ft_amp, 'plasma', vmin, vmax), (ph, 'twilight', -np.pi, np.pi),
               (sim_ph, 'twilight', -np.pi, np.pi), (ft_ph, 'twilight', -np.pi, np.pi)]
        for j, spec in enumerate(ims):
            if j == 1:
                ax[r, j].imshow(dt.T, aspect='auto', origin='lower', extent=ext, vmin=vmin, vmax=vmax, cmap='plasma')
                ax[r, j].imshow(green(fl), aspect='auto', origin='lower', extent=ext)
            else:
                img, cm, lo, hi = spec
                ax[r, j].imshow(img.T, aspect='auto', origin='lower', extent=ext, vmin=lo, vmax=hi, cmap=cm)
            ax[r, j].tick_params(labelsize=6); ax[r, j].set_xlabel("time", fontsize=7)
        ax[r, 0].set_ylabel(f"baseline {idx}\nFreq MHz\nflag={fl.mean():.2f}", fontsize=7)
        if r == 0:
            for j, t in enumerate(titles):
                ax[r, j].set_title(t, fontsize=8)

    plt.tight_layout()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"-> {out}", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--sim-ckpt', required=True, dest='sim_ckpt')
    ap.add_argument('--ft-ckpt', required=True, dest='ft_ckpt')
    ap.add_argument('--output', required=True)
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
