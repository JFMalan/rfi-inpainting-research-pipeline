import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import phase1, phase2
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, psnr, phase_error, complex_mae, noise_floor_ratio, to_ampphase


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


@torch.no_grad()
def val_eval(diff, ema_model, val_dl, cfg, out, epoch):
    ema_model.eval()
    seen = 0
    maes, psnrs, pherrs, cplxs, nfrs = [], [], [], [], []
    mf_maes, mf_cplxs = [], []
    first = None
    vr = getattr(cfg, 'vis_repr', 'ampphase')
    clip = (-4.0, 4.0) if vr == 'realimag' else (-2.0, 4.0)
    for batch in val_dl:
        x0 = batch['clean'].to(diff.device)
        mask = batch['mask'].to(diff.device)
        cond = build_cond(batch['corrupted'].to(diff.device), mask, batch['pe'].to(diff.device),
                          hole_fill=getattr(cfg, 'hole_fill', 'mean'), vis_repr=vr)
        pred = diff.sample(ema_model, cond, x0, mask, predict=cfg.predict, eta=0.0,
                           steps=cfg.val_eval_steps, clip=clip)
        pc, xc = to_ampphase(pred, vr), to_ampphase(x0, vr)   # canonical amp+cos+sin for scoring
        maes.append(float(mae(pc, xc, mask)))
        psnrs.append(float(psnr(pc, xc, mask)))
        if xc.shape[1] >= 3:
            pherrs.append(float(phase_error(pc, xc, mask)))
            cplxs.append(float(complex_mae(pc, xc, mask)))
        else:
            pherrs.append(0.0)
            cplxs.append(float(mae(pc, xc, mask)))
        nfrs.append(float(noise_floor_ratio(pc, xc, mask)))
        # mean-fill baseline: per-patch mean of known pixels, all channels
        base = x0.clone()
        keep = mask == 0
        for i in range(x0.shape[0]):
            for c in range(x0.shape[1]):
                base[i, c] = x0[i, c][keep[i, 0]].mean()
        bc = to_ampphase(base, vr)
        mf_maes.append(float(mae(bc, xc, mask)))
        mf_cplxs.append(float(complex_mae(bc, xc, mask)))
        if first is None:
            first = (x0.cpu().numpy(), batch['corrupted'].numpy(),
                     batch['mask'].numpy(), pred.cpu().numpy(),
                     batch['fmin'].numpy(), batch['fmax'].numpy())
        seen += x0.shape[0]
        if seen >= cfg.val_eval_patches:
            break
    np.savez(out / f'sample_e{epoch}.npz',
             clean=first[0], corrupted=first[1], mask=first[2], pred=first[3],
             fmin=first[4], fmax=first[5])
    return {'amp_mae': float(np.mean(maes)), 'psnr': float(np.mean(psnrs)),
            'phase_err': float(np.mean(pherrs)), 'complex_mae': float(np.mean(cplxs)),
            'noise_floor_ratio': float(np.mean(nfrs)),
            'mf_amp_mae': float(np.mean(mf_maes)), 'mf_complex_mae': float(np.mean(mf_cplxs))}


def main(args):
    extra = {k: v for k, v in dict(lr=args.lr, val_eval_steps=args.val_eval_steps,
                                   val_eval_patches=args.val_eval_patches).items()
             if v is not None}
    if args.amp_only:
        extra['amp_only'] = True
        extra['target_channels'] = 1
    if args.vis_repr == 'realimag':
        extra['vis_repr'] = 'realimag'
        extra['target_channels'] = 2
    if args.rand_mask:
        extra['rand_mask'] = True
    if args.raw_amp:
        extra['raw_amp'] = True
    if args.loss:
        extra['loss_kind'] = args.loss
    cfg = (phase2 if args.phase == 2 else phase1)(
        data_glob=args.data, out_dir=args.out, epochs=args.epochs,
        batch_size=args.batch_size, max_patches=args.max_patches, seed=args.seed,
        smooth_target=args.smooth_target, clean_target=args.clean_target, **extra,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'samples').mkdir(exist_ok=True)

    ds = PatchDataset(cfg.data_glob, pe_channels=cfg.pe_channels,
                      augment=cfg.augment, max_patches=cfg.max_patches, split='train',
                      smooth_target=cfg.smooth_target, smooth_sigma=cfg.smooth_sigma,
                      clean_target=cfg.clean_target, amp_only=cfg.amp_only,
                      rand_mask=cfg.rand_mask, raw_amp=cfg.raw_amp, vis_repr=cfg.vis_repr)
    val_ds = PatchDataset(cfg.data_glob, pe_channels=cfg.pe_channels,
                          augment=False, split='val',
                          smooth_target=cfg.smooth_target, smooth_sigma=cfg.smooth_sigma,
                          clean_target=cfg.clean_target, amp_only=cfg.amp_only,
                          raw_amp=cfg.raw_amp, vis_repr=cfg.vis_repr)
    print(f"dataset: train {len(ds)}  val {len(val_ds)}  {ds.n_time}x{ds.n_freq}  device={device}")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M  in_channels={cfg.in_channels}")

    diff = Diffusion(T=cfg.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    ema = EMA(model, cfg.ema_decay)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        ema.shadow.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        start_epoch = ck['epoch'] + 1
        print(f"resumed from epoch {start_epoch}")

    best_cplx = ck['best_cplx'] if args.resume and Path(args.resume).exists() and 'best_cplx' in ck else 1e9
    stale = 0
    log = []
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for batch in dl:
            opt.zero_grad()
            loss = diff.loss(model, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)
            running += loss.item()
        sched.step()
        avg = running / len(dl)
        dt = time.time() - t0
        line = {'epoch': epoch, 'loss': round(avg, 5), 'sec': round(dt, 1),
                'lr': round(opt.param_groups[0]['lr'], 6)}

        evaluated = (epoch + 1) % cfg.sample_every == 0 or epoch == cfg.epochs - 1
        if evaluated:
            v = val_eval(diff, ema.shadow, val_dl, cfg, out / 'samples', epoch)
            # headline + baselines on one line so the log shows "beating baseline?" at a glance
            line['complex_mae'] = round(v['complex_mae'], 5)
            line['complex_mf'] = round(v['mf_complex_mae'], 5)
            line['amp_mae'] = round(v['amp_mae'], 5)
            line['amp_mf'] = round(v['mf_amp_mae'], 5)
            line['psnr'] = round(v['psnr'], 3)
            line['phase_err'] = round(v['phase_err'], 4)
            line['nfr'] = round(v['noise_floor_ratio'], 3)
            line['beats_mf'] = bool(v['complex_mae'] < v['mf_complex_mae'])

        print(json.dumps(line), flush=True)
        log.append(line)
        (out / 'log.json').write_text(json.dumps(log, indent=2))

        state = {'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
                 'opt': opt.state_dict(), 'epoch': epoch, 'best_cplx': best_cplx,
                 'cfg': vars(cfg)}
        if (epoch + 1) % cfg.ckpt_every == 0 or epoch == cfg.epochs - 1:
            torch.save(state, out / 'ckpt.pt')

        if evaluated:
            c = v['complex_mae']
            improved = c < best_cplx - cfg.min_delta
            if c < best_cplx:
                best_cplx = c
                state['best_cplx'] = best_cplx
                torch.save(state, out / 'best.pt')
                print(f"  new best complex_mae {c:.5f} -> best.pt", flush=True)
            stale = 0 if improved else stale + 1
            if cfg.early_stop and epoch + 1 >= cfg.min_epochs and stale >= cfg.patience:
                print(f"early stop: no >{cfg.min_delta} complex-MAE gain for {stale} evals "
                      f"(best {best_cplx:.5f})", flush=True)
                break

    ds.close()
    print("done")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--phase', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-patches', type=int, default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--val-eval-steps', type=int, default=None)
    ap.add_argument('--val-eval-patches', type=int, default=None)
    ap.add_argument('--smooth-target', action='store_true', dest='smooth_target')
    ap.add_argument('--clean-target', action='store_true', dest='clean_target')
    ap.add_argument('--amp-only', action='store_true', dest='amp_only')
    ap.add_argument('--rand-mask', action='store_true', dest='rand_mask')
    ap.add_argument('--raw-amp', action='store_true', dest='raw_amp')
    ap.add_argument('--loss', default=None, choices=['l1', 'l2', 'l1l2'])
    ap.add_argument('--repr', default='ampphase', choices=['ampphase', 'realimag'], dest='vis_repr')
    main(ap.parse_args())
