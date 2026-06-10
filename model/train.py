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
from metrics import mae, psnr


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
    maes, psnrs = [], []
    first = None
    for batch in val_dl:
        x0 = batch['clean'].to(diff.device)
        mask = batch['mask'].to(diff.device)
        cond = build_cond(batch['corrupted'].to(diff.device), mask, batch['pe'].to(diff.device))
        pred = diff.sample(ema_model, cond, x0, mask, predict=cfg.predict)
        maes.append(float(mae(pred, x0, mask)))
        psnrs.append(float(psnr(pred, x0, mask)))
        if first is None:
            first = (x0.cpu().numpy(), batch['corrupted'].numpy(),
                     batch['mask'].numpy(), pred.cpu().numpy())
        seen += x0.shape[0]
        if seen >= cfg.val_eval_patches:
            break
    np.savez(out / f'sample_e{epoch}.npz',
             clean=first[0], corrupted=first[1], mask=first[2], pred=first[3])
    return float(np.mean(maes)), float(np.mean(psnrs))


def main(args):
    cfg = (phase2 if args.phase == 2 else phase1)(
        data_glob=args.data, out_dir=args.out, epochs=args.epochs,
        batch_size=args.batch_size, max_patches=args.max_patches, seed=args.seed,
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'samples').mkdir(exist_ok=True)

    ds = PatchDataset(cfg.data_glob, pe_channels=cfg.pe_channels,
                      augment=cfg.augment, max_patches=cfg.max_patches, split='train')
    val_ds = PatchDataset(cfg.data_glob, pe_channels=cfg.pe_channels,
                          augment=False, split='val')
    print(f"dataset: train {len(ds)}  val {len(val_ds)}  {ds.n_time}x{ds.n_freq}  device={device}")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    model = UNet(cfg.in_channels, out_ch=1, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M  in_channels={cfg.in_channels}")

    diff = Diffusion(T=cfg.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ema = EMA(model, cfg.ema_decay)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        ema.shadow.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        start_epoch = ck['epoch'] + 1
        print(f"resumed from epoch {start_epoch}")

    best_psnr = ck['best_psnr'] if args.resume and Path(args.resume).exists() and 'best_psnr' in ck else -1e9
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
        avg = running / len(dl)
        dt = time.time() - t0
        line = {'epoch': epoch, 'loss': avg, 'sec': round(dt, 1)}

        evaluated = (epoch + 1) % cfg.sample_every == 0 or epoch == cfg.epochs - 1
        if evaluated:
            m, p = val_eval(diff, ema.shadow, val_dl, cfg, out / 'samples', epoch)
            line['mae'] = round(m, 5)
            line['psnr'] = round(p, 3)

        print(json.dumps(line), flush=True)
        log.append(line)
        (out / 'log.json').write_text(json.dumps(log, indent=2))

        state = {'model': model.state_dict(), 'ema': ema.shadow.state_dict(),
                 'opt': opt.state_dict(), 'epoch': epoch, 'best_psnr': best_psnr,
                 'cfg': vars(cfg)}
        if (epoch + 1) % cfg.ckpt_every == 0 or epoch == cfg.epochs - 1:
            torch.save(state, out / 'ckpt.pt')

        if evaluated:
            improved = p > best_psnr + cfg.min_delta
            if p > best_psnr:
                best_psnr = p
                state['best_psnr'] = best_psnr
                torch.save(state, out / 'best.pt')
                print(f"  new best psnr {p:.3f} -> best.pt", flush=True)
            stale = 0 if improved else stale + 1
            if cfg.early_stop and epoch + 1 >= cfg.min_epochs and stale >= cfg.patience:
                print(f"early stop: no >{cfg.min_delta}dB gain for {stale} evals "
                      f"(best psnr {best_psnr:.3f})", flush=True)
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
    main(ap.parse_args())
