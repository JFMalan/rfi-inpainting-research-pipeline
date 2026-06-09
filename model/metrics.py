import torch


def mae(pred, target, mask=None):
    if mask is None:
        return (pred - target).abs().mean()
    region = mask > 0
    if region.sum() == 0:
        return torch.tensor(0.0)
    return (pred - target).abs()[region].mean()


def psnr(pred, target, mask=None, data_range=None):
    if data_range is None:
        data_range = target.max() - target.min()
    if mask is None:
        mse = ((pred - target) ** 2).mean()
    else:
        region = mask > 0
        if region.sum() == 0:
            return torch.tensor(0.0)
        mse = ((pred - target) ** 2)[region].mean()
    return 20 * torch.log10(data_range / torch.sqrt(mse + 1e-12))


def tre(pred, dirty, mask):
    # Phase 2 / real data — no clean ground truth.
    # Total Reconstruction Error per Luo et al.: error built from the binary RFI
    # mask, the dirty spectrogram, the prediction, and the gradient magnitude of
    # the reconstruction. Implement against real-data conventions in phase 2.
    raise NotImplementedError("TRE is implemented in phase 2 (real observational data)")
