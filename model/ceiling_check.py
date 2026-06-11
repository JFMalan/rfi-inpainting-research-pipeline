import argparse

import numpy as np


def psnr(pred, clean, region, data_range):
    mse = ((pred - clean) ** 2)[region].mean()
    return 20 * np.log10(data_range / np.sqrt(mse + 1e-12))


def main(args):
    d = np.load(args.input)
    clean, pred, mask = d['clean'], d['pred'], d['mask']
    # amplitude is channel 0 for 3-channel arrays
    if clean.ndim == 4 and clean.shape[1] >= 3:
        clean, pred = clean[:, 0], pred[:, 0]
    else:
        clean = clean.squeeze(1) if clean.ndim == 4 else clean
        pred = pred.squeeze(1) if pred.ndim == 4 else pred
    mask = mask.squeeze(1) if mask.ndim == 4 else mask

    region = mask > 0
    dr = clean.max() - clean.min()

    model_psnr, meanfill_psnr, noisefloor_psnr = [], [], []
    for i in range(len(clean)):
        r = region[i]
        if r.sum() < 10:
            continue
        known = clean[i][~r]
        mu, sd = known.mean(), known.std()
        meanfill = np.full_like(clean[i], mu)
        # mean-fill: predict the local mean everywhere in the hole
        meanfill_psnr.append(psnr(meanfill, clean[i], r, dr))
        # noise-floor: best possible if you also knew the noise std but not the realisation
        rng = np.random.default_rng(i)
        noisefill = mu + sd * rng.standard_normal(clean[i].shape).astype(np.float32)
        noisefloor_psnr.append(psnr(noisefill, clean[i], r, dr))
        model_psnr.append(psnr(pred[i], clean[i], r, dr))

    print(f"patches scored      : {len(model_psnr)}")
    print(f"clean amp std       : {clean.std():.4f}  (data range {dr:.3f})")
    print(f"MODEL    psnr (mask): {np.mean(model_psnr):.3f}  +/- {np.std(model_psnr):.3f}")
    print(f"mean-fill psnr (mask): {np.mean(meanfill_psnr):.3f}  +/- {np.std(meanfill_psnr):.3f}")
    print(f"noise-fill psnr(mask): {np.mean(noisefloor_psnr):.3f}  +/- {np.std(noisefloor_psnr):.3f}")
    print()
    gain = np.mean(model_psnr) - np.mean(meanfill_psnr)
    print(f"model - mean-fill   : {gain:+.3f} dB")
    if gain < 0.3:
        print("=> model ~= mean-fill: amplitude is at the noise ceiling (data lacks structure).")
    elif gain > 1.0:
        print("=> model clearly beats mean-fill: it IS reconstructing structure; headroom exists.")
    else:
        print("=> model modestly beats mean-fill: some structure learned, limited by data.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    main(ap.parse_args())
