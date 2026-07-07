import argparse

import h5py
import numpy as np


def main(args):
    hf = h5py.File(args.h5, 'r')
    preds = np.load(args.preds)['preds']
    n = min(args.max_units or preds.shape[0], preds.shape[0], hf['mask'].shape[0])
    amp_bias = []; amp_mae = []; ph_mae = []; cplx_rel = []; frac = []
    for u in range(n):
        m = hf['mask'][u] > 0.5
        if m.sum() < 5:
            continue
        clean = hf['clean'][u]; ph = hf['phase'][u]; div = hf['dn_divisor'][u]
        ap = preds[u, 0]; pp = np.arctan2(preds[u, 2], preds[u, 1])
        Vt = clean * div * np.exp(1j * ph)
        Vp = ap * div * np.exp(1j * pp)
        amp_bias.append(float((ap[m] - clean[m]).mean()))          # signed -> systematic offset
        amp_mae.append(float(np.abs(ap[m] - clean[m]).mean()))
        dphi = np.angle(np.exp(1j * (pp[m] - ph[m])))
        ph_mae.append(float(np.abs(dphi).mean()))
        cplx_rel.append(float(np.abs(Vp[m] - Vt[m]).mean() / (np.abs(Vt[m]).mean() + 1e-12)))
        frac.append(float(m.mean()))
    hf.close()
    print(f"units scored: {len(amp_mae)}  mean hole frac {np.mean(frac):.3f}")
    print(f"  amplitude (divisively-normalised units):")
    print(f"    mean signed bias  {np.mean(amp_bias):+.4f}   (systematic offset -> coherent image artefacts)")
    print(f"    mean |error|      {np.mean(amp_mae):.4f}")
    print(f"  phase:")
    print(f"    mean |error|      {np.degrees(np.mean(ph_mae)):.2f} deg")
    print(f"  complex visibility:")
    print(f"    mean |Vpred-Vtrue| / mean|Vtrue|  = {np.mean(cplx_rel):.3f}   (relative reconstruction error)")
    print(f"  read: small random error but a nonzero signed bias -> continuum rings; "
          f"large |error| everywhere -> model/scale bug")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True, help='dataset_wN.h5 (has clean/phase/dn_divisor/mask)')
    ap.add_argument('--preds', required=True, help='preds_wN.npz')
    ap.add_argument('--max-units', type=int, default=400, dest='max_units')
    main(ap.parse_args())
