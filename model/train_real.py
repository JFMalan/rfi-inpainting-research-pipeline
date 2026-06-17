import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import phase2
from data import RealDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, tre


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
    tres, fid_maes, mf_maes = [], [], []
    first = None
    for batch in val_dl:
        obs = batch['obs'].to(diff.device)
        hidden = batch['hidden'].to(diff.device)
        fake = batch['fake_mask'].to(diff.device)
        cond = build_cond(obs, hidden, batch['pe'].to(diff.device),
                          hole_fill=getattr(cfg, 'hole_fill', 'mean'))
        # sample with both holes hidden; recover the fill, score only on fake holes
        pred = diff.sample(ema_model, cond, obs, hidden, predict=cfg.predict,
                           eta=0.0, steps=200)
        tres.append(float(tre(pred, obs, fake)))
        fid_maes.append(float(mae(pred, obs, fake)))
        base = obs.clone()
        keep = hidden == 0
        for i in range(obs.shape[0]):
            for c in range(obs.shape[1]):
                base[i, c] = obs[i, c][keep[i, 0]].mean()
        mf_maes.append(float(mae(base, obs, fake)))
        if first is None:
            first = (obs.cpu().numpy(), batch['real_flags'].numpy(),
                     batch['fake_mask'].numpy(), pred.cpu().numpy(),
                     batch['fmin'].numpy(), batch['fmax'].numpy())
        seen += obs.shape[0]
        if seen >= cfg.val_eval_patches:
            break
    np.savez(out / f'sample_e{epoch}.npz',
             obs=first[0], real_flags=first[1], fake_mask=first[2], pred=first[3],
             fmin=first[4], fmax=first[5])
    return {'tre': float(np.mean(tres)), 'fake_mae': float(np.mean(fid_maes)),
            'mf_fake_mae': float(np.mean(mf_maes))}


def main(args):
    cfg = phase2(data_glob=args.data, out_dir=args.out, epochs=args.epochs,
                 batch_size=args.batch_size, max_patches=args.max_patches, seed=args.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'samples').mkdir(exist_ok=True)

    ds = RealDataset(cfg.data_glob, pe_channels=cfg.pe_channels, augment=cfg.augment,
                     max_patches=cfg.max_patches, split='train',
                     fake_mask_frac=cfg.fake_mask_frac)
    val_ds = RealDataset(cfg.data_glob, pe_channels=cfg.pe_channels, augment=False,
                         split='val', fake_mask_frac=cfg.fake_mask_frac)
    print(f"device={device} ({gpu})  train {len(ds)}  val {len(val_ds)}  "
          f"{ds.n_time}x{ds.n_freq}  init={'scratch' if not args.init_from else args.init_from}",
          flush=True)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        sd = ck['ema'] if 'ema' in ck else ck['model']
        model.load_state_dict(sd)
        print(f"initialised from sim checkpoint {args.init_from} (ema weights)", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M  in_channels={cfg.in_channels}", flush=True)

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
        print(f"resumed from epoch {start_epoch}", flush=True)

    best_tre = 1e9
    stale = 0
    log = []
    total_iters = 0
    hit_cap = False
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        nit = 0
        for it, batch in enumerate(dl):
            opt.zero_grad()
            loss = diff.loss_phase2(model, batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)
            running += loss.item()
            nit += 1
            total_iters += 1
            if it == 0 or (it + 1) % 50 == 0:
                rate = (it + 1) / max(time.time() - t0, 1e-6)
                print(f"  e{epoch} it{it+1} (tot {total_iters}) loss {loss.item():.4f} "
                      f"({rate:.2f} it/s)", flush=True)
            if args.max_iters and total_iters >= args.max_iters:
                hit_cap = True
                break
        sched.step()
        avg = running / max(nit, 1)
        dt = time.time() - t0
        line = {'epoch': epoch, 'loss': round(avg, 5), 'sec': round(dt, 1),
                'lr': round(opt.param_groups[0]['lr'], 6)}

        evaluated = (epoch + 1) % cfg.sample_every == 0 or epoch == cfg.epochs - 1 or hit_cap
        if evaluated:
            v = val_eval(diff, ema.shadow, val_dl, cfg, out / 'samples', epoch)
            line['tre'] = round(v['tre'], 5)
            line['fake_mae'] = round(v['fake_mae'], 5)
            line['mf_fake_mae'] = round(v['mf_fake_mae'], 5)
            line['beats_mf'] = bool(v['fake_mae'] < v['mf_fake_mae'])

        print(json.dumps(line), flush=True)
        log.append(line)
        (out / 'log.json').write_text(json.dumps(log, indent=2))

        state = {'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
                 'opt': opt.state_dict(), 'epoch': epoch, 'best_tre': best_tre,
                 'cfg': vars(cfg)}
        if (epoch + 1) % cfg.ckpt_every == 0 or epoch == cfg.epochs - 1:
            torch.save(state, out / 'ckpt.pt')

        if evaluated:
            c = v['tre']
            improved = c < best_tre - cfg.min_delta
            if c < best_tre:
                best_tre = c
                state['best_tre'] = best_tre
                torch.save(state, out / 'best.pt')
                print(f"  new best tre {c:.5f} -> best.pt", flush=True)
            stale = 0 if improved else stale + 1
            if cfg.early_stop and epoch + 1 >= cfg.min_epochs and stale >= cfg.patience:
                print(f"early stop: no >{cfg.min_delta} TRE gain for {stale} evals "
                      f"(best {best_tre:.5f})", flush=True)
                break
        if hit_cap:
            print(f"reached max_iters={args.max_iters}; stopping", flush=True)
            break

    ds.close()
    print("done", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--init-from', default=None)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-patches', type=int, default=None)
    ap.add_argument('--max-iters', type=int, default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())
