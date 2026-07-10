import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))

from config import phase1
from data import PatchDataset, build_cond
from diffusion import Diffusion
from unet import UNet


def hdr(s):
    print(f"\n{'='*64}\n{s}\n{'='*64}")


def amp_mae(p, t, r):
    return float(np.abs(p - t)[r].mean())


def interp_fill(a, h):
    out = a.copy(); nt, nf = a.shape; idx = np.arange(nf)
    for tt in range(nt):
        hr = h[tt]
        if hr.any() and not hr.all():
            out[tt, hr] = np.interp(idx[hr], idx[~hr], a[tt, ~hr])
    return out


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = phase1()
    torch.manual_seed(0); np.random.seed(0)
    ds = PatchDataset(args.data, pe_channels=cfg.pe_channels, augment=False,
                      split='train', max_patches=args.n)
    print(f"device={dev}  patches={len(ds)}  in_channels={cfg.in_channels}  predict={cfg.predict}")

    # ---- TEST 1: DATA INTEGRITY ---------------------------------------------
    hdr("TEST 1  data integrity (is structure present, context real?)")
    s = ds[0]
    clean = s['clean'].numpy(); corr = s['corrupted'].numpy(); m = s['mask'].numpy()
    amp_c, amp_x = clean[0], corr[0]; mask2 = m[0] > 0
    print(f"  shapes clean{clean.shape} corrupted{corr.shape} mask{m.shape}")
    print(f"  mask fraction {mask2.mean():.3f}")
    # outside the mask, corrupted should equal clean (context is the true signal)
    out_match = np.allclose(amp_c[~mask2], amp_x[~mask2], atol=1e-4)
    print(f"  [{'PASS' if out_match else 'FAIL'}] corrupted==clean OUTSIDE mask (context is real signal)")
    # inside the mask, corrupted != clean (RFI was injected) -> there IS a hole to fill
    rfi_in = float(np.abs(amp_x[mask2] - amp_c[mask2]).mean()) if mask2.any() else 0
    print(f"  RFI magnitude inside mask: {rfi_in:.3f}  (should be >0)")
    # is there recoverable structure? interp vs mean-fill on the clean amplitude
    mu = amp_c[~mask2].mean()
    mf = amp_mae(np.full_like(amp_c, mu), amp_c, mask2)
    ip = amp_mae(interp_fill(amp_c, mask2), amp_c, mask2)
    print(f"  amp recoverability: mean-fill {mf:.3f}  interp {ip:.3f}  "
          f"[{'structure recoverable' if ip < mf-0.01 else 'NO recoverable structure'}]")

    # ---- TEST 2: CONDITIONING DELIVERY --------------------------------------
    hdr("TEST 2  conditioning delivers real context into the network input")
    batch = {k: torch.stack([ds[i][k] for i in range(2)]) for k in s}
    cond = build_cond(batch['corrupted'], batch['mask'], batch['pe'], hole_fill=cfg.hole_fill)
    mk = batch['mask'][:, 0] > 0
    known_ch0 = cond[:, 0]  # conditioning amplitude channel
    # OUTSIDE the mask the cond amplitude should equal the true clean amplitude
    ctx_ok = torch.allclose(known_ch0[~mk], batch['clean'][:, 0][~mk], atol=1e-4)
    print(f"  cond channels: {cond.shape[1]} (expect {cfg.target_channels+1+cfg.pe_channels})")
    print(f"  [{'PASS' if ctx_ok else 'FAIL'}] conditioning carries TRUE amplitude outside mask")
    # INSIDE the mask, cond should be the fill (mean), not the true value (no leak)
    inside_filled = float(known_ch0[mk].mean())
    print(f"  cond amplitude INSIDE mask mean: {inside_filled:.3f} (hole_fill={cfg.hole_fill}; should NOT equal true structure)")

    # ---- TEST 3: U-NET INFORMATION FLOW known->hole -------------------------
    hdr("TEST 3  can the U-Net propagate known pixels INTO the hole? (receptive field)")
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base,
                 ch_mult=cfg.ch_mult, attn_res=cfg.attn_res, num_res=cfg.num_res,
                 img_size=cfg.img_size).to(dev)
    model.eval()
    x = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size, device=dev)
    t = torch.zeros(1, device=dev, dtype=torch.long)
    y0 = model(x, t)
    # perturb ONE known pixel far from centre; does the centre output change?
    x2 = x.clone(); x2[0, :, 10, 10] += 5.0
    y1 = model(x2, t)
    delta_centre = (y1 - y0)[0, 0, 128, 128].abs().item()
    delta_near = (y1 - y0)[0, 0, 20, 20].abs().item()
    print(f"  perturb pixel (10,10): output change at (20,20)={delta_near:.4e}  at centre(128,128)={delta_centre:.4e}")
    print(f"  [{'PASS' if delta_centre > 1e-6 else 'FAIL'}] info propagates across the patch (global receptive field)")

    # ---- TEST 4: TRAINING CONTRACT (x_in) -----------------------------------
    hdr("TEST 4  training input contract: clean known + hidden hole")
    diff = Diffusion(T=cfg.timesteps, device=dev)
    x0 = batch['clean'].to(dev); mm = batch['mask'].to(dev)
    noise = torch.randn_like(x0); tt = torch.full((2,), 500, device=dev, dtype=torch.long)
    xt = diff.q_sample(x0, tt, noise)
    keep = 1.0 - mm
    x_in = keep * x0 + mm * xt
    leak = torch.allclose((x_in * mm)[mm.expand_as(x_in) > 0],
                          (x0 * mm)[mm.expand_as(x0) > 0], atol=1e-3)
    known_clean = torch.allclose((x_in * keep)[keep.expand_as(x_in) > 0],
                                 (x0 * keep)[keep.expand_as(x0) > 0], atol=1e-4)
    print(f"  [{'PASS' if known_clean else 'FAIL'}] known region of x_in == clean x0")
    print(f"  [{'PASS' if not leak else 'FAIL'}] hole of x_in is NOISED (not raw truth -> no trivial leak)")

    # ---- TEST 5: LOSS GRADES THE HOLE, RIGHT TARGET -------------------------
    hdr("TEST 5  loss is in the hole, against the right target, per channel")
    loss = diff.loss(model, {k: batch[k].to(dev) for k in batch}, cfg)
    print(f"  loss value: {float(loss):.4f}  finite={torch.isfinite(loss).item()}")
    # a model outputting the TRUE x0 should give ~0 loss; outputting zeros should give large loss
    class Truth(torch.nn.Module):
        def __init__(self, x0): super().__init__(); self.x0 = x0
        def forward(self, x, t): return self.x0
    truth_loss = diff.loss(Truth(x0), {k: batch[k].to(dev) for k in batch}, cfg)
    print(f"  [{'PASS' if float(truth_loss) < 0.05 else 'FAIL'}] perfect-x0 model -> ~0 loss ({float(truth_loss):.4f})")

    # ---- TEST 6: CAN IT OVERFIT 2 PATCHES (capacity + flow end-to-end) ------
    hdr("TEST 6  overfit 2 patches leak-free (can the WHOLE chain learn at all?)")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    model.train()
    tb = {k: batch[k].to(dev) for k in batch}
    for it in range(args.iters):
        opt.zero_grad(); L = diff.loss(model, tb, cfg); L.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = diff.sample(model, build_cond(tb['corrupted'], tb['mask'], tb['pe'], hole_fill=cfg.hole_fill),
                           tb['clean'], tb['mask'], predict=cfg.predict, eta=0.0, steps=200)
    r = (tb['mask'][:, 0] > 0).cpu().numpy()
    a_pred = pred[:, 0].cpu().numpy(); a_true = tb['clean'][:, 0].cpu().numpy()
    mae_model = float(np.abs(a_pred - a_true)[r].mean())
    mae_mf = np.mean([amp_mae(np.full_like(a_true[i], a_true[i][~r[i]].mean()), a_true[i], r[i])
                      for i in range(len(a_true))])
    print(f"  after {args.iters} iters: amp MAE model {mae_model:.4f}  mean-fill {mae_mf:.4f}")
    print(f"  [{'PASS' if mae_model < mae_mf else 'FAIL'}] model beats mean-fill on 2 memorized patches")
    print(f"  final train loss {float(L):.4f}")

    print("\nDONE. Read the FAIL lines above to localise the issue.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--iters', type=int, default=1500)
    main(ap.parse_args())
