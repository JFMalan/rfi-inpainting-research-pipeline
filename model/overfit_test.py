import argparse

import numpy as np
import torch

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, psnr, phase_error


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase1(predict=args.predict)
    cfg.hole_fill = args.hole_fill
    if args.amp_only:
        cfg.target_channels = 1
    torch.manual_seed(0)
    print(f"predict mode: {cfg.predict}  amp_only: {args.amp_only}  in_channels: {cfg.in_channels}")

    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False,
                      split='train', max_patches=args.n, amp_only=args.amp_only)
    full = {k: torch.stack([ds[i][k] for i in range(args.n)]) for k in ds[0]}
    bs = min(args.bs, args.n)
    print(f"overfitting {args.n} patches on {dev}  (batch {bs})")

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size).to(dev)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    g = torch.Generator().manual_seed(0)
    model.train()
    for it in range(args.iters):
        idx = torch.randint(0, args.n, (bs,), generator=g)
        mb = {k: full[k][idx].to(dev) for k in full}
        opt.zero_grad()
        loss = diff.loss(model, mb, cfg)
        loss.backward()
        opt.step()
        if (it + 1) % 50 == 0:
            print(f"  iter {it+1}/{args.iters}  loss {loss.item():.4f}", flush=True)

    # in-mask noise-prediction error vs t (the decisive signal)
    model.eval()
    ne = min(args.n, args.eval_n)
    print(f"\nin-mask |pred-eps| by channel vs t (eval on {ne} patches):")
    with torch.no_grad():
        x0 = full['clean'][:ne].to(dev); m = full['mask'][:ne].to(dev)
        cond = build_cond(full['corrupted'][:ne].to(dev), m, full['pe'][:ne].to(dev),
                          hole_fill=cfg.hole_fill)
        for tv in [800, 500, 200, 50]:
            t = torch.full((x0.shape[0],), tv, device=dev, dtype=torch.long)
            eps = torch.randn_like(x0)
            xt = diff.q_sample(x0, t, eps)
            pred = model(torch.cat([xt, cond], dim=1), t)
            err = (pred - eps).abs()
            inm = m > 0
            chans = "  ".join(f"ch{c}:{err[:,c:c+1][inm].mean().item():.3f}" for c in range(x0.shape[1]))
            print(f"  t={tv:4d}  {chans}")

        rmask = m > 0
        amp_true = x0[:, 0:1]

        # (A) single-shot x0_pred from a mid-level noised state: the conditional-mean
        # estimate, NO stochastic sampling. should be smooth like interp if structure learned.
        t = torch.full((x0.shape[0],), 100, device=dev, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = diff.q_sample(x0, t, eps)
        x_in = (m == 0).float() * xt + (m > 0).float() * xt  # full xt; cond carries known
        if cfg.predict == 'x0':
            x0_pred = model(torch.cat([xt, cond], dim=1), t).clamp(-2, 4)
        else:
            x0_pred, _ = diff.predict_x0(model, xt, cond, t, clip=(-2, 4))
        x0pred_mae = (x0_pred[:, 0:1] - amp_true).abs()[rmask].mean().item()

        # (B) full stochastic sample (the actual inpainting output)
        pred = diff.sample(model, cond, x0, m, predict=cfg.predict, U=args.U)
        amp_pred = pred[:, 0:1]
        model_mae = (amp_pred - amp_true).abs()[rmask].mean().item()
        # mean-fill baseline (per-patch local mean of known pixels)
        mf = torch.zeros_like(amp_true)
        interp = amp_true.clone()
        for i in range(x0.shape[0]):
            known = amp_true[i][m[i] == 0]
            mf[i] = known.mean()
            # per-row linear interp across freq (the classical recoverable target)
            a = amp_true[i, 0].cpu().numpy()
            h = (m[i, 0] > 0).cpu().numpy()
            nt, nf2 = a.shape
            idx = np.arange(nf2)
            for tt in range(nt):
                hr = h[tt]
                if hr.any() and not hr.all():
                    a[tt, hr] = np.interp(idx[hr], idx[~hr], a[tt, ~hr])
            interp[i, 0] = torch.from_numpy(a).to(dev)
        meanfill_mae = (mf - amp_true).abs()[rmask].mean().item()
        interp_mae = (interp - amp_true).abs()[rmask].mean().item()
        p_model = float(psnr(pred, x0, m))
        ph = float(phase_error(pred, x0, m))
    print(f"\nMASK-REGION MAE (physical units, lower=better):")
    print(f"  MODEL (sampled)    : {model_mae:.4f}")
    print(f"  MODEL (x0_pred)    : {x0pred_mae:.4f}   <- conditional mean, no stochastic noise")
    print(f"  interp (target)    : {interp_mae:.4f}   <- classical recoverable structure")
    print(f"  mean-fill baseline : {meanfill_mae:.4f}")
    print(f"  phase_err          : {ph:.3f} rad")
    print(f"  PSNR (secondary)   : {p_model:.2f} dB")
    print("VERDICT:", "PASS (x0_pred beats mean-fill -> structure learned)"
          if x0pred_mae < meanfill_mae - 0.005
          else "FAIL (model ~= mean-fill, not recovering structure)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--iters', type=int, default=600)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--eval-n', type=int, default=8, dest='eval_n')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--predict', default='noise', choices=['noise', 'x0'])
    ap.add_argument('--amp-only', action='store_true', dest='amp_only')
    ap.add_argument('--hole-fill', default='mean', choices=['zero', 'mean', 'noise', 'center'], dest='hole_fill')
    ap.add_argument('--U', type=int, default=1)
    main(ap.parse_args())
