import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, psnr, phase_error, complex_mae


def texture_ratio(pred_amp, clean_amp, mask):
    ratios = []
    for b in range(pred_amp.shape[0]):
        m = mask[b] > 0
        if m.sum() < 20 or (~m).sum() < 20:
            continue
        hp_p = pred_amp[b] - uniform_filter(pred_amp[b], size=5, mode='nearest')
        hp_k = clean_amp[b] - uniform_filter(clean_amp[b], size=5, mode='nearest')
        if hp_k[~m].std() > 1e-6:
            ratios.append(hp_p[m].std() / hp_k[~m].std())
    return float(np.mean(ratios)) if ratios else 0.0


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'
    print(f"device={dev} ({gpu_name})", flush=True)
    cfg = phase1(predict=args.predict)
    print("loading dataset", flush=True)
    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False, split='val')
    batch = {k: torch.stack([ds[i][k] for i in range(args.n)]).to(dev) for k in ds[0]}
    x0, m = batch['clean'], batch['mask']
    cond = build_cond(batch['corrupted'], m, batch['pe'], hole_fill=cfg.hole_fill)
    print(f"loaded {args.n} patches, building model", flush=True)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    print("model loaded, starting sweep", flush=True)

    configs = [(eta, u) for eta in args.etas for u in args.repaint_u]
    cl = x0[:, 0].cpu().numpy(); mk = m[:, 0].cpu().numpy()
    saved = {}
    print(f"predict={cfg.predict}  steps={args.steps}  patches={args.n}")
    print(f"{'eta':>5} {'U':>3} | {'amp_mae':>8} {'cplx':>8} {'psnr':>7} {'phase':>7} {'TEXTURE':>8}")
    print("-" * 56)
    for eta, u in configs:
        t0 = time.time()

        def prog(k, total, _t0=t0, _eta=eta, _u=u):
            if k % 100 == 0 or k == total:
                el = time.time() - _t0
                print(f"  [eta{_eta} u{_u}] step {k}/{total}  {el:.1f}s  {k / max(el, 1e-6):.1f} step/s",
                      flush=True)

        pred = diff.sample(model, cond, x0, m, predict=cfg.predict, clip=tuple(args.clip),
                           eta=eta, steps=args.steps, repaint_u=u, progress=prog)
        amp_mae = float(mae(pred, x0, m))
        cplx = float(complex_mae(pred, x0, m))
        ps = float(psnr(pred, x0, m))
        ph = float(phase_error(pred, x0, m))
        tex = texture_ratio(pred[:, 0].cpu().numpy(), cl, mk)
        key = f"eta{eta}_u{u}"
        saved[f"pred_{key}"] = pred.cpu().numpy()
        print(f"{eta:>5.2f} {u:>3d} | {amp_mae:>8.4f} {cplx:>8.4f} {ps:>7.2f} {ph:>7.3f} {tex:>8.3f}")

    mf_amp = x0[:, 0:1].clone()
    for i in range(x0.shape[0]):
        mf_amp[i] = x0[i, 0:1][m[i, 0:1] == 0].mean()
    print(f"\nmean-fill baseline texture (sanity, should be ~0): "
          f"{texture_ratio(mf_amp[:, 0].cpu().numpy(), cl, mk):.3f}")
    print("(texture 1.0 = fill speckle matches surroundings; <1 too smooth; >1 too noisy)")

    np.savez(args.out, clean=x0.cpu().numpy(), corrupted=batch['corrupted'].cpu().numpy(),
             mask=m.cpu().numpy(), fmin=batch['fmin'].cpu().numpy(),
             fmax=batch['fmax'].cpu().numpy(), **saved)
    print(f"saved -> {args.out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--predict', default='x0', choices=['noise', 'x0'])
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--etas', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    ap.add_argument('--repaint-u', type=int, nargs='+', default=[1, 5], dest='repaint_u')
    ap.add_argument('--clip', type=float, nargs=2, default=[-2.0, 4.0])
    main(ap.parse_args())
