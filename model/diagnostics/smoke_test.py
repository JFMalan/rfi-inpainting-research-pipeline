import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet
from metrics import mae, psnr, phase_error


def main(args):
    cfg = phase1()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={dev}  in_channels={cfg.in_channels}  target_channels={cfg.target_channels}")

    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=True, split='train')
    print(f"train patches: {len(ds)}  {ds.n_time}x{ds.n_freq}")
    assert ds.n_time == cfg.img_size and ds.n_freq == cfg.img_size, \
        f"patch shape {ds.n_time}x{ds.n_freq} != img_size {cfg.img_size}"

    s = ds[0]
    print("sample shapes:", {k: tuple(v.shape) for k, v in s.items() if hasattr(v, 'shape')})
    assert s['clean'].shape[0] == cfg.target_channels
    assert s['corrupted'].shape[0] == cfg.target_channels

    batch = {k: torch.stack([ds[i][k] for i in range(2)]) for k in s}
    cond = build_cond(batch['corrupted'], batch['mask'], batch['pe'])
    print("cond channels:", cond.shape[1], "(expect", cfg.target_channels + 1 + cfg.pe_channels, ")")

    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size).to(dev)
    diff = Diffusion(T=cfg.timesteps, device=dev)
    for k in batch:
        batch[k] = batch[k].to(dev)

    loss = diff.loss(model, batch, cfg)
    print("forward + loss OK:", float(loss))
    loss.backward()
    print("backward OK")

    diff_small = Diffusion(T=5, device=dev)
    cond = build_cond(batch['corrupted'], batch['mask'], batch['pe'])
    with torch.no_grad():
        pred = diff_small.sample(model, cond, batch['clean'], batch['mask'], predict=cfg.predict)
    print("sample OK:", tuple(pred.shape),
          "mae", float(mae(pred, batch['clean'], batch['mask'])),
          "phase_err", float(phase_error(pred, batch['clean'], batch['mask'])))
    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    main(ap.parse_args())
