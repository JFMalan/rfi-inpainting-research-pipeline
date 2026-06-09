import torch
import torch.nn.functional as F

from data import build_cond


def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    ac = torch.cos(((steps / T + s) / (1 + s)) * torch.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return betas.clamp(1e-8, 0.999).float()


class Diffusion:
    def __init__(self, T=1000, device='cpu'):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, dim=0)
        self.acp_prev = F.pad(self.acp[:-1], (1, 0), value=1.0)
        self.sqrt_acp = torch.sqrt(self.acp)
        self.sqrt_one_minus_acp = torch.sqrt(1 - self.acp)
        self.post_var = betas * (1 - self.acp_prev) / (1 - self.acp)

    def _gather(self, a, t, shape):
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise):
        return (self._gather(self.sqrt_acp, t, x0.shape) * x0
                + self._gather(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def loss(self, model, batch, cfg):
        x0 = batch['clean'].to(self.device)
        cond = build_cond(batch['corrupted'].to(self.device),
                          batch['mask'].to(self.device),
                          batch['pe'].to(self.device))
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(torch.cat([xt, cond], dim=1), t)

        target = noise if cfg.predict == 'noise' else x0
        m = batch['mask'].to(self.device)
        region = m if cfg.loss_region == 'mask' else torch.ones_like(m)

        err = (pred - target).abs()
        denom = region.sum().clamp(min=1.0)
        masked = (err * region).sum() / denom
        glob = err.mean()
        return cfg.mask_weight * masked + (1 - cfg.mask_weight) * glob

    def predict_x0(self, model, xt, cond, t):
        pred = model(torch.cat([xt, cond], dim=1), t)
        sqrt_acp = self._gather(self.sqrt_acp, t, xt.shape)
        sqrt_omacp = self._gather(self.sqrt_one_minus_acp, t, xt.shape)
        return (xt - sqrt_omacp * pred) / sqrt_acp, pred

    @torch.no_grad()
    def sample(self, model, cond, x0_known, mask, predict='noise'):
        device = self.device
        shape = x0_known.shape
        x = torch.randn(shape, device=device)
        keep = (mask == 0).float()
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            if predict == 'noise':
                x0_pred, eps = self.predict_x0(model, x, cond, t)
            else:
                x0_pred = model(torch.cat([x, cond], dim=1), t)
                eps = (x - self._gather(self.sqrt_acp, t, shape) * x0_pred) \
                    / self._gather(self.sqrt_one_minus_acp, t, shape)
            mean = (self._gather(torch.sqrt(self.acp_prev), t, shape) * x0_pred
                    + self._gather(torch.sqrt(1 - self.acp_prev - self.post_var), t, shape) * eps)
            if i > 0:
                noise = torch.randn_like(x)
                x_unknown = mean + torch.sqrt(self._gather(self.post_var, t, shape)) * noise
                x_known = self.q_sample(x0_known, t, torch.randn_like(x))
            else:
                x_unknown = mean
                x_known = x0_known
            x = keep * x_known + (1 - keep) * x_unknown
        return x
