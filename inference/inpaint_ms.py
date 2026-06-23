import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from casacore.tables import table, maketabledesc, makecoldesc
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'model'))

from config import phase1, phase2
from data import positional_encoding, build_cond, smooth_component
from diffusion import Diffusion
from unet import UNet

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def resize_native(arr, n_time, n_chan, order=1, aa=False):
    return resize(arr, (n_time, n_chan), order=order, mode='edge',
                  anti_aliasing=aa, preserve_range=True).astype(np.float32)


def infer(diff, model, cfg, data, phase, hole, band_min, band_max, sz, dev,
          steps, nf, smooth_tgt, smooth_sigma):
    amp = smooth_component(data, hole, smooth_sigma) if smooth_tgt else data
    obs = np.stack([amp, np.cos(phase), np.sin(phase)], 0)[None]
    x0 = torch.from_numpy(obs).to(dev)
    m = torch.from_numpy(hole[None, None].astype(np.float32)).to(dev)
    pe = positional_encoding(band_min, band_max, band_min, band_max, sz, sz, cfg.pe_channels)
    pe = torch.from_numpy(pe[None].copy()).to(dev)
    cond = build_cond(x0, m, pe, hole_fill=getattr(cfg, 'hole_fill', 'mean'))
    pred = diff.sample(model, cond, x0, m, predict=cfg.predict, eta=0.0, steps=steps, noise_floor=nf)
    return pred[0].cpu().numpy()   # (3, sz, sz) = amp_norm, cos, sin


def ensure_column(root, out_col, src_col, chunk=50000):
    if out_col in root.colnames():
        log(f"column {out_col} exists, reusing")
        return
    log(f"adding {out_col} (copy of {src_col})")
    desc = root.getcoldesc(src_col)
    root.addcols(maketabledesc(makecoldesc(out_col, desc)))
    n = root.nrows()
    for s in range(0, n, chunk):
        nr = min(chunk, n - s)
        root.putcol(out_col, root.getcol(src_col, startrow=s, nrow=nr), startrow=s, nrow=nr)
    log(f"initialised {out_col} from {src_col} ({n} rows)")


def main(args):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'
    amp_key = 'corrupted' if args.sim else 'data'
    hole_key = 'mask' if args.sim else 'flags'
    nf = None if args.noise_floor in (None, 'none') else (
        'auto' if args.noise_floor == 'auto' else float(args.noise_floor))
    log(f"device={dev} ({gpu})  ms={args.ms}  h5={args.h5}  out_col={args.out_col}  "
        f"noise_floor={nf}  unflag={args.unflag}")

    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    chan_lo = int(hf.attrs['chan_lo'])
    full_n_chan = int(hf.attrs['full_n_chan'])
    chan_hi = chan_lo + full_n_chan
    band_min = float(hf.attrs['freq_min_mhz'])
    band_max = float(hf.attrs['freq_max_mhz'])
    n_units = hf[amp_key].shape[0]
    has_tlo = 'time_lo' in hf
    log(f"h5: {n_units} units  sz={sz}  chan {chan_lo}:{chan_hi}  band {band_min:.1f}-{band_max:.1f} MHz")

    root = table(args.ms, readonly=False, ack=False)
    src_col = args.src_col if args.src_col in root.colnames() else 'DATA'

    times = root.getcol('TIME')
    n_row = root.nrows()
    n_time = len(np.unique(times))
    if n_row % n_time != 0:
        raise RuntimeError(f"n_row {n_row} not divisible by n_time {n_time} — non-rectangular MS, row map unsafe")
    n_baseline = n_row // n_time
    # time-major ordering check: each block of n_baseline rows must share one timestamp
    tblock = times[:n_time * n_baseline].reshape(n_time, n_baseline)
    if not np.all(tblock.max(axis=1) == tblock.min(axis=1)):
        raise RuntimeError("MS rows are not time-major (baseline order varies per time) — row map unsafe")
    log(f"MS: {n_row} rows  n_time={n_time}  n_baseline={n_baseline}  src_col={src_col}")

    if args.field is not None:
        ensure_column(root, args.out_col, src_col)
        ms = root.query(f"FIELD_ID == {args.field}")
        # recompute the selection geometry (extraction used the same query)
        st = ms.getcol('TIME'); n_row = ms.nrows(); n_time = len(np.unique(st))
        n_baseline = n_row // n_time
        log(f"field {args.field} selection: {n_row} rows  n_time={n_time}  n_baseline={n_baseline}")
    else:
        ensure_column(root, args.out_col, src_col)
        ms = root

    cfg = (phase1 if args.sim else phase2)(predict=args.predict)
    model = UNet(cfg.in_channels, out_ch=cfg.target_channels, base=cfg.base, ch_mult=cfg.ch_mult,
                 attn_res=cfg.attn_res, num_res=cfg.num_res, img_size=cfg.img_size).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    model.eval()
    diff = Diffusion(T=cfg.timesteps, device=dev)
    log("model loaded")

    cap = n_units if args.max_units is None else min(args.max_units, n_units)
    written = 0
    with torch.no_grad():
        for u in range(cap):
            data = hf[amp_key][u].astype(np.float32)
            phase = hf['phase'][u].astype(np.float32)
            hole = hf[hole_key][u].astype(np.float32)
            divisor = hf['dn_divisor'][u].astype(np.float32)
            bl = int(hf['baseline_id'][u])
            nt = int(hf['native_n_time'][u])
            nc = int(hf['native_n_chan'][u])
            tlo = int(hf['time_lo'][u]) if has_tlo else 0
            if nc != full_n_chan:
                raise RuntimeError(f"unit {u}: native_n_chan {nc} != full_n_chan {full_n_chan}")

            pred = infer(diff, model, cfg, data, phase, hole, band_min, band_max, sz, dev,
                         args.steps, nf, args.smooth_target, args.smooth_sigma)

            amp_n = resize_native(pred[0], nt, nc)
            div_n = resize_native(divisor, nt, nc)
            cos_n = resize_native(pred[1], nt, nc)
            sin_n = resize_native(pred[2], nt, nc)
            hole_n = resize_native(hole, nt, nc) > 0.5
            ang = np.arctan2(sin_n, cos_n)
            V = (amp_n * div_n * np.exp(1j * ang)).astype(np.complex64)   # (nt, nc)

            sr = tlo * n_baseline + bl
            d = ms.getcol(args.out_col, startrow=sr, nrow=nt, rowincr=n_baseline)  # (nt, nchan_tot, npol)
            npol = d.shape[2]
            band = d[:, chan_lo:chan_hi, :]
            for p in range(npol):
                band[:, :, p] = np.where(hole_n, V, band[:, :, p])
            ms.putcol(args.out_col, d, startrow=sr, nrow=nt, rowincr=n_baseline)

            if args.unflag:
                fl = ms.getcol('FLAG', startrow=sr, nrow=nt, rowincr=n_baseline)
                fb = fl[:, chan_lo:chan_hi, :]
                for p in range(npol):
                    fb[:, :, p] = np.where(hole_n, False, fb[:, :, p])
                ms.putcol('FLAG', fl, startrow=sr, nrow=nt, rowincr=n_baseline)

            written += 1
            if written == 1 or written % 100 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                log(f"  wrote unit {written}/{cap}  bl={bl} tlo={tlo} holes={int(hole_n.sum())}  ({rate:.2f}/s)")

    ms.flush(); root.flush()
    hf.close()
    log(f"done: inpainted {written} units into {args.out_col}"
        f"{' and cleared FLAG in holes' if args.unflag else ''}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True, help='extracted dataset with metadata (divisor, baseline_id, ...)')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-col', default='INPAINTED_DATA', dest='out_col')
    ap.add_argument('--src-col', default='DATA', dest='src_col')
    ap.add_argument('--field', type=int, default=None)
    ap.add_argument('--predict', default='x0')
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--noise-floor', default='auto', dest='noise_floor', help='none | auto | float')
    ap.add_argument('--unflag', action='store_true', help='clear FLAG at inpainted pixels')
    ap.add_argument('--sim', action='store_true', help='sim dataset.h5 (keys corrupted/mask) instead of real (data/flags)')
    ap.add_argument('--smooth-target', action='store_true', dest='smooth_target')
    ap.add_argument('--smooth-sigma', type=float, default=1.0, dest='smooth_sigma')
    ap.add_argument('--max-units', type=int, default=None, dest='max_units')
    main(ap.parse_args())
