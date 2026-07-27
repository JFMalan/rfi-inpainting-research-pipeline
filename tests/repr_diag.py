import argparse
import time

import h5py
import numpy as np
from casacore.tables import table
from skimage.transform import resize

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


def to512(arr, sz):
    return resize(arr, (sz, sz), order=1, mode='edge', anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


def to_native(arr, nt, nc):
    return resize(arr, (nt, nc), order=1, mode='edge', anti_aliasing=False,
                  preserve_range=True).astype(np.float32)


def wrap(a):
    return np.angle(np.exp(1j * a))


def main(args):
    hole_key = 'mask' if args.sim else 'flags'
    amp_key = 'clean' if args.sim else 'data'
    hf = h5py.File(args.h5, 'r')
    sz = int(hf.attrs['img_size'])
    chan_lo = int(hf.attrs['chan_lo'])
    has_tlo = 'time_lo' in hf
    has_flo = 'freq_lo' in hf

    root = table(args.ms, readonly=True, ack=False)
    times = root.getcol('TIME')
    n_time = len(np.unique(times))
    n_baseline = root.nrows() // n_time

    n_units = hf[hole_key].shape[0]
    cap = n_units if args.max_units is None else min(args.max_units, n_units)
    log(f"diag {cap}/{n_units} units  n_baseline={n_baseline}  sz={sz}")

    acc = {k: 0.0 for k in ('amp', 'ph_h5', 'ph_fix', 'full_h5', 'full_fix', 'ref', 'pol', 'dat')}
    deg_h5 = []; deg_fix = []
    npix = 0

    for u in range(cap):
        bl = int(hf['baseline_id'][u]); nt = int(hf['native_n_time'][u]); nc = int(hf['native_n_chan'][u])
        tlo = int(hf['time_lo'][u]) if has_tlo else 0
        flo = int(hf['freq_lo'][u]) if has_flo else 0
        clo = chan_lo + flo; chi = clo + nc
        sr = tlo * n_baseline + bl

        D = root.getcol('DATA', startrow=sr, nrow=nt, rowincr=n_baseline)[:, clo:chi, :]
        hole = resize(hf[hole_key][u].astype(np.float32), (nt, nc), order=0, mode='edge',
                      preserve_range=True) > 0.5
        if hole.sum() == 0:
            continue

        Dm = D.mean(axis=2)
        theta = np.angle(Dm).astype(np.float32)
        amp_col = np.abs(D).mean(axis=2).astype(np.float32)
        Vref = amp_col * np.exp(1j * theta)

        amp_recon = to_native(hf[amp_key][u], nt, nc) * to_native(hf['dn_divisor'][u], nt, nc)

        ph512 = hf['phase'][u]
        ph_h5 = np.arctan2(to_native(np.sin(ph512), nt, nc), to_native(np.cos(ph512), nt, nc))
        ph_fix = np.arctan2(to_native(to512(np.sin(theta), sz), nt, nc),
                            to_native(to512(np.cos(theta), sz), nt, nc))

        h = hole
        V_amp = amp_recon * np.exp(1j * theta)
        V_ph_h5 = amp_col * np.exp(1j * ph_h5)
        V_ph_fix = amp_col * np.exp(1j * ph_fix)
        V_full_h5 = amp_recon * np.exp(1j * ph_h5)
        V_full_fix = amp_recon * np.exp(1j * ph_fix)

        acc['ref'] += np.sum(np.abs(Vref[h]) ** 2)
        acc['amp'] += np.sum(np.abs(V_amp[h] - Vref[h]) ** 2)
        acc['ph_h5'] += np.sum(np.abs(V_ph_h5[h] - Vref[h]) ** 2)
        acc['ph_fix'] += np.sum(np.abs(V_ph_fix[h] - Vref[h]) ** 2)
        acc['full_h5'] += np.sum(np.abs(V_full_h5[h] - Vref[h]) ** 2)
        acc['full_fix'] += np.sum(np.abs(V_full_fix[h] - Vref[h]) ** 2)

        Vref_b = np.repeat(Vref[:, :, None], D.shape[2], axis=2)
        acc['pol'] += np.sum(np.abs(Vref_b[h] - D[h]) ** 2)
        acc['dat'] += np.sum(np.abs(D[h]) ** 2)

        deg_h5.append(np.degrees(np.sqrt(np.mean(wrap(ph_h5[h] - theta[h]) ** 2))))
        deg_fix.append(np.degrees(np.sqrt(np.mean(wrap(ph_fix[h] - theta[h]) ** 2))))
        npix += int(h.sum())

        if u == 0 or (u + 1) % 50 == 0:
            log(f"  unit {u + 1}/{cap}  holes={int(h.sum())}  cum_pix={npix}")

    root.close(); hf.close()
    ref = np.sqrt(acc['ref'])
    dat = np.sqrt(acc['dat'])
    log("=== visibility-domain error in holes (relative to perfect pol-mean Vref) ===")
    log(f"  amp resize roundtrip      {np.sqrt(acc['amp']) / ref:.4f}")
    log(f"  phase h5-angle (current)  {np.sqrt(acc['ph_h5']) / ref:.4f}")
    log(f"  phase cos/sin FIX         {np.sqrt(acc['ph_fix']) / ref:.4f}")
    log(f"  full pipeline (current)   {np.sqrt(acc['full_h5']) / ref:.4f}")
    log(f"  full pipeline + phase FIX {np.sqrt(acc['full_fix']) / ref:.4f}")
    log(f"  pol-collapse (one V/pol)  {np.sqrt(acc['pol']) / dat:.4f}  (vs true per-pol DATA)")
    log("=== phase error in holes (RMS degrees vs native angle) ===")
    log(f"  h5-angle  {np.mean(deg_h5):.2f} deg   cos/sin FIX  {np.mean(deg_fix):.2f} deg")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', required=True)
    ap.add_argument('--h5', required=True)
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--max-units', type=int, default=200, dest='max_units')
    main(ap.parse_args())
