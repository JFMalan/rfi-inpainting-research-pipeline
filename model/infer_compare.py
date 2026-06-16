import argparse

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, complex_mae


def add_calibrated_noise(pred, x0_known, mask, scale=1.0):
    # add Gaussian noise to the hole calibrated so the fill's high-pass speckle std
    # matches the known region. Adding raw std overshoots the high-pass measure
    # (white noise is all high-freq), so an empirical `scale` tunes it to texture~1.
    out = pred.clone()
    for b in range(pred.shape[0]):
        m = mask[b, 0] > 0
        if m.sum() < 5 or (~m).sum() < 20:
            continue
        c = x0_known[b, 0]
        hp_known = (c - torch.nn.functional.avg_pool2d(c[None, None], 5, 1, 2)[0, 0])[~m]
        sd = hp_known.std() * scale
        out[b, 0][m] = out[b, 0][m] + sd * torch.randn(int(m.sum()), device=pred.device)
    return out


def texture_ratio(pred_amp, clean_amp, mask):
    # high-freq (speckle) std in the hole vs the known region. ~1.0 = texture matches;
    # <1 = fill too smooth; >1 = too noisy. Computed on the high-pass residual.
    ratios = []
    for b in range(pred_amp.shape[0]):
        m = mask[b] > 0
        if m.sum() < 20 or (~m).sum() < 20:
            continue
        p = pred_amp[b]; c = clean_amp[b]
        hp_pred = p - uniform_filter(p, size=5, mode='nearest')   # fill high-freq
        hp_known = c - uniform_filter(c, size=5, mode='nearest')  # true high-freq
        s_hole = hp_pred[m].std()
        s_known = hp_known[~m].std()
        if s_known > 1e-6:
            ratios.append(s_hole / s_known)
    return float(np.mean(ratios)) if ratios else 0.0


def run(diff, model, cond, x0, m, predict, clip, steps, eta=0.0, repaint_u=1):
    pred = diff.sample(model, cond, x0, m, predict=predict, clip=clip, eta=eta,
                       steps=steps, repaint_u=repaint_u)
    return float(mae(pred, x0, m)), float(complex_mae(pred, x0, m)), pred


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase1()
    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='val')
    n = args.n
    batch = {k: torch.stack([ds[i][k] for i in range(n)]).to(dev) for k in ds[0]}
    x0, m = batch['clean'], batch['mask']
    cond = build_cond(batch['corrupted'], m, batch['pe'], hole_fill=cfg.hole_fill)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema']); model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)

    # data-driven tight clamp from the known region
    known = x0[:, 0][m[:, 0] == 0]
    lo, hi = float(known.mean() - 5 * known.std()), float(known.mean() + 5 * known.std())
    print(f"data range (known amp): mean {known.mean():.3f} std {known.std():.3f}  tight clip ({lo:.2f},{hi:.2f})")

    # compare current DDIM (repaint_u=1) vs RePaint resampling (u=5, u=10),
    # with WIDE vs NARROW mask error split.
    def wide_narrow(pred):
        we, ne = [], []
        for b in range(pred.shape[0]):
            mm = m[b, 0] > 0
            wide_cols = mm.float().mean(dim=0) > 0.5
            ap, at = pred[b, 0], x0[b, 0]
            wp = mm & wide_cols[None, :]; npx = mm & ~wide_cols[None, :]
            if wp.any(): we.append((ap - at).abs()[wp].mean().item())
            if npx.any(): ne.append((ap - at).abs()[npx].mean().item())
        return (float(np.mean(we)) if we else 0.0), (float(np.mean(ne)) if ne else 0.0)

    # judge by TEXTURE MATCH (does the fill have the speckle of the surroundings?)
    # NOT just MAE (which rewards smooth means). texture ~1.0 is the goal.
    cl_np = x0[:, 0].cpu().numpy(); m_np = m[:, 0].cpu().numpy()
    print("  (texture: 1.0 = fill matches surrounding speckle; <1 too smooth)")
    amp0, cx0, pred_mean = run(diff, model, cond, x0, m, cfg.predict, (-2.0, 4.0), 1000, eta=0.0)
    tex0 = texture_ratio(pred_mean[:, 0].cpu().numpy(), cl_np, m_np)
    print(f"  mean (eta0)            :  amp_mae {amp0:.4f}  complex {cx0:.4f}  TEXTURE {tex0:.3f}")
    saved = {'mean': pred_mean.cpu().numpy()}
    best = None
    for scale in [0.4, 0.55, 0.7]:
        pn = add_calibrated_noise(pred_mean, x0, m, scale=scale)
        tex = texture_ratio(pn[:, 0].cpu().numpy(), cl_np, m_np)
        amn = float(mae(pn, x0, m))
        print(f"  mean+noise scale={scale} :  amp_mae {amn:.4f}  TEXTURE {tex:.3f}")
        saved[f'noise{scale}'] = pn.cpu().numpy()
        if best is None or abs(tex - 1.0) < abs(best[1] - 1.0):
            best = (scale, tex)
    print(f"  -> best texture-match scale={best[0]} (texture {best[1]:.3f})")

    np.savez(args.out, clean=x0.cpu().numpy(), corrupted=batch['corrupted'].cpu().numpy(),
             mask=m.cpu().numpy(), pred_mean=saved['mean'],
             pred_n04=saved['noise0.4'], pred_n055=saved['noise0.55'], pred_n07=saved['noise0.7'],
             fmin=batch['fmin'].cpu().numpy(), fmax=batch['fmax'].cpu().numpy())
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6)
    main(ap.parse_args())
