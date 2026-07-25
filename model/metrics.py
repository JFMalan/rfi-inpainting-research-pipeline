import torch
import torch.nn.functional as F


def to_ampphase(x, vis_repr='ampphase'):
    # convert a batch to the canonical amplitude+cos+sin representation so all metrics
    # (and both representations) are scored identically. real/imag -> amp, cos, sin.
    if vis_repr != 'realimag' or x.shape[1] < 2:
        return x
    re, im = x[:, 0:1], x[:, 1:2]
    amp = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    ang = torch.atan2(im, re)
    return torch.cat([amp, torch.cos(ang), torch.sin(ang)], dim=1)


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


def _highpass(x):
    pad = F.pad(x, (2, 2, 2, 2), mode='reflect')
    return x - F.avg_pool2d(pad, 5, stride=1)


def noise_floor_ratio(pred, target, mask, flags=None):
    # texture consistency: std of the 5x5 high-pass residual inside the hole (pred)
    # vs the surrounding known region (target). 1.0 = inpaint carries the same noise
    # floor as the data around it; ~0 = over-smooth fill. amplitude channel only.
    # flags (real RFI) are excluded from the known region so junk doesn't inflate it.
    p, t = _amp(pred), _amp(target)
    hp_p, hp_t = _highpass(p), _highpass(t)
    hole = mask > 0
    known = ~hole
    if flags is not None:
        known = known & ~(flags > 0)
    ratios = []
    for b in range(p.shape[0]):
        hb, kb = hole[b, 0], known[b, 0]
        if hb.sum() < 20 or kb.sum() < 20:
            continue
        t_std = hp_t[b, 0][kb].std()
        if t_std > 1e-6:
            ratios.append(hp_p[b, 0][hb].std() / t_std)
    if not ratios:
        return torch.tensor(0.0)
    return torch.stack(ratios).mean()


def _grad_mag(x):
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def tre(pred, dirty, mask, flags=None, lam=1.0):
    # Total Reconstruction Error (Luo et al.) for real data with no clean target.
    # Two terms, both on amplitude:
    #   fidelity  — prediction must agree with the observed (dirty) data OUTSIDE the
    #               mask, where the observation is trusted. RFI-flagged pixels (flags)
    #               are NOT trusted, so they are excluded from fidelity even though
    #               they sit outside the fake hole.
    #   roughness — gradient magnitude of the reconstruction INSIDE the mask; a
    #               seamless fill has low boundary/interior roughness.
    p, d = _amp(pred), _amp(dirty)
    region = mask > 0
    keep = ~region
    if flags is not None:
        keep = keep & ~(flags > 0)
    if keep.sum() == 0 or region.sum() == 0:
        return torch.tensor(0.0)
    fidelity = (p - d).abs()[keep].mean()
    roughness = _grad_mag(p)[region].mean()
    return fidelity + lam * roughness
