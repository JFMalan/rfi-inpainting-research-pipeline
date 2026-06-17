import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    cfg = phase1(predict=args.predict)
    print(f"device={dev} ({gpu})  img_size={cfg.img_size}  ch_mult={cfg.ch_mult}  "
          f"attn_res={cfg.attn_res}  in_channels={cfg.in_channels}", flush=True)

    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False,
                      split='train', max_patches=max(args.batches))
    sample = ds[0]
    print(f"dataset {len(ds)} baselines  shapes: "
          f"clean {tuple(sample['clean'].shape)}  mask {tuple(sample['mask'].shape)}  "
          f"pe {tuple(sample['pe'].shape)}", flush=True)

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {n_params/1e6:.1f}M params", flush=True)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def make_batch(bs):
        items = [ds[i % len(ds)] for i in range(bs)]
        return {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}

    print(f"\n{'batch':>6} {'fwd+bwd s/it':>14} {'peak GB':>9} {'status':>8}", flush=True)
    results = []
    for bs in args.batches:
        try:
            if dev == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            batch = make_batch(bs)
            model.train()
            # warmup + timed iters
            for w in range(2):
                opt.zero_grad()
                loss = diff.loss(model, batch, cfg)
                loss.backward()
                opt.step()
            if dev == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.iters):
                opt.zero_grad()
                loss = diff.loss(model, batch, cfg)
                loss.backward()
                opt.step()
            if dev == 'cuda':
                torch.cuda.synchronize()
            dt = (time.time() - t0) / args.iters
            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0.0
            print(f"{bs:>6} {dt:>14.3f} {peak:>9.2f} {'ok':>8}", flush=True)
            results.append((bs, dt, peak))
            del batch
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"{bs:>6} {'-':>14} {'-':>9} {'OOM':>8}", flush=True)
                if dev == 'cuda':
                    torch.cuda.empty_cache()
                break
            raise

    if results:
        bs, dt, peak = results[-1]
        print(f"\nlargest fitting batch: {bs}  ({dt:.3f} s/it, {peak:.1f} GB)", flush=True)
        for nsamp in args.epoch_sizes:
            iters = nsamp / bs
            per_epoch = iters * dt
            print(f"  {nsamp} samples @ batch {bs}: {iters:.0f} it/epoch, "
                  f"~{per_epoch/60:.1f} min/epoch, ~{per_epoch*30/3600:.1f} h for 30 epochs",
                  flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--batches', type=int, nargs='+', default=[2, 4, 8, 12, 16])
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--predict', default='x0', choices=['noise', 'x0'])
    ap.add_argument('--epoch-sizes', type=int, nargs='+', default=[2016, 10000], dest='epoch_sizes')
    main(ap.parse_args())
