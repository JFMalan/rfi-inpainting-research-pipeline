import argparse

import numpy as np
from scipy.ndimage import uniform_filter


def mae(pred, clean, region):
    return np.abs(pred - clean)[region].mean()


def psnr(pred, clean, region, dr):
    mse = ((pred - clean) ** 2)[region].mean()
    return 20 * np.log10(dr / np.sqrt(mse + 1e-12))


def main(args):
    d = np.load(args.input)
    clean, pred, mask = d['clean'], d['pred'], d['mask']
    if clean.ndim == 4 and clean.shape[1] >= 3:
        clean, pred = clean[:, 0], pred[:, 0]
    else:
        clean = clean.squeeze(1) if clean.ndim == 4 else clean
        pred = pred.squeeze(1) if pred.ndim == 4 else pred
    mask = mask.squeeze(1) if mask.ndim == 4 else mask

    region = mask > 0
    dr = clean.max() - clean.min()

    model_mae, meanfill_mae, noisefloor_mae = [], [], []
    model_psnr = []
    struct_mae, struct_meanfill = [], []
    for i in range(len(clean)):
        r = region[i]
        if r.sum() < 10:
            continue
        known = clean[i][~r]
        mu = known.mean()
        meanfill = np.full_like(clean[i], mu)

        model_mae.append(mae(pred[i], clean[i], r))
        meanfill_mae.append(mae(meanfill, clean[i], r))
        model_psnr.append(psnr(pred[i], clean[i], r, dr))

        # noise floor: split clean into smooth structure + high-freq residual.
        # a perfect inpainter recovers structure but never the per-pixel residual,
        # so the achievable MAE in the hole is ~ the residual's local scale.
        smooth = uniform_filter(clean[i], size=8, mode='nearest')
        resid = clean[i] - smooth
        noise_sd = resid[~r].std()
        # best-case predictor = true smooth structure; residual stays as error
        noisefloor_mae.append(mae(smooth, clean[i], r))
        # structure-only comparison: how well does the model match the SMOOTH part?
        smooth_pred = uniform_filter(pred[i], size=8, mode='nearest')
        struct_mae.append(np.abs(smooth_pred - smooth)[r].mean())
        struct_meanfill.append(np.abs(mu - smooth)[r].mean())

    print(f"patches scored        : {len(model_mae)}")
    print(f"clean amp std         : {clean.std():.4f}   data range {dr:.3f}")
    print()
    print("MASK-REGION MAE (physical units, lower=better):")
    print(f"  MODEL               : {np.mean(model_mae):.4f}  +/- {np.std(model_mae):.4f}")
    print(f"  mean-fill baseline  : {np.mean(meanfill_mae):.4f}")
    print(f"  noise-floor (best)  : {np.mean(noisefloor_mae):.4f}   <- irreducible, perfect-structure target")
    print()
    print("STRUCTURE-ONLY MAE (smooth component, the learnable part):")
    print(f"  MODEL structure     : {np.mean(struct_mae):.4f}")
    print(f"  mean-fill structure : {np.mean(struct_meanfill):.4f}")
    print()
    print(f"  MODEL psnr (mask)   : {np.mean(model_psnr):.3f} dB   (secondary, for paper comparison)")
    print()

    full_gain = np.mean(meanfill_mae) - np.mean(model_mae)       # >0 means better than mean-fill
    floor_gap = np.mean(model_mae) - np.mean(noisefloor_mae)     # how far above the achievable floor
    struct_gain = np.mean(struct_meanfill) - np.mean(struct_mae)
    print(f"model vs mean-fill (MAE reduction): {full_gain:+.4f}")
    print(f"model above noise floor          : {floor_gap:+.4f}")
    print(f"structure recovery vs mean-fill  : {struct_gain:+.4f}")
    if struct_gain > 0.01 and np.mean(model_mae) < np.mean(meanfill_mae) - 0.005:
        print("=> model recovers real structure (beats mean-fill on the learnable part).")
    elif np.mean(model_mae) >= np.mean(meanfill_mae) - 0.005:
        print("=> model ~= mean-fill: not recovering structure (undertrained or limited).")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    main(ap.parse_args())
