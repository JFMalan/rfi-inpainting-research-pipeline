import torch


def _amp(x):
    # channel 0 is amplitude when multi-channel (amp, cos, sin); else use as-is
    return x[:, 0:1] if x.dim() == 4 and x.shape[1] > 1 else x


def mae(pred, target, mask=None):
    p, t = _amp(pred), _amp(target)
    if mask is None:
        return (p - t).abs().mean()
    region = mask > 0
    if region.sum() == 0:
        return torch.tensor(0.0)
    return (p - t).abs()[region].mean()


def psnr(pred, target, mask=None, data_range=None):
    p, t = _amp(pred), _amp(target)
    if data_range is None:
        data_range = t.max() - t.min()
    if mask is None:
        mse = ((p - t) ** 2).mean()
    else:
        region = mask > 0
        if region.sum() == 0:
            return torch.tensor(0.0)
        mse = ((p - t) ** 2)[region].mean()
    return 20 * torch.log10(data_range / torch.sqrt(mse + 1e-12))


def phase_error(pred, target, mask=None):
    # mean absolute angular error (radians) from cos/sin channels (1=cos, 2=sin)
    if pred.shape[1] < 3:
        return torch.tensor(0.0)
    ang_p = torch.atan2(pred[:, 2:3], pred[:, 1:2])
    ang_t = torch.atan2(target[:, 2:3], target[:, 1:2])
    d = torch.atan2(torch.sin(ang_p - ang_t), torch.cos(ang_p - ang_t)).abs()
    if mask is None:
        return d.mean()
    region = mask > 0
    if region.sum() == 0:
        return torch.tensor(0.0)
    return d[region].mean()


def complex_mae(pred, target, mask=None, divisor=None):
    # error on the recombined complex visibility V = amp*(cos + i*sin).
    # this is the actual reconstructed quantity. divisor (dn_divisor) optionally
    # de-normalises amplitude back to physical Jy before measuring.
    if pred.shape[1] < 3:
        return torch.tensor(0.0)
    ap, at = pred[:, 0:1], target[:, 0:1]
    if divisor is not None:
        ap, at = ap * divisor, at * divisor
    vp_re, vp_im = ap * pred[:, 1:2], ap * pred[:, 2:3]
    vt_re, vt_im = at * target[:, 1:2], at * target[:, 2:3]
    err = torch.sqrt((vp_re - vt_re) ** 2 + (vp_im - vt_im) ** 2 + 1e-12)
    if mask is None:
        return err.mean()
    region = mask > 0
    if region.sum() == 0:
        return torch.tensor(0.0)
    return err[region].mean()


def tre(pred, dirty, mask):
    # Phase 2 / real data — no clean ground truth.
    # Total Reconstruction Error per Luo et al.: error built from the binary RFI
    # mask, the dirty spectrogram, the prediction, and the gradient magnitude of
    # the reconstruction. Implement against real-data conventions in phase 2.
    raise NotImplementedError("TRE is implemented in phase 2 (real observational data)")
