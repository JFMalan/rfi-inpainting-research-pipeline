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
        # DDPM posterior mean coefficients (Ho et al. 2020 eq. 7):
        # mean = c_x0 * x0_pred + c_xt * x_t
        self.post_c_x0 = torch.sqrt(self.acp_prev) * betas / (1 - self.acp)
        self.post_c_xt = torch.sqrt(self.alphas) * (1 - self.acp_prev) / (1 - self.acp)

    def _gather(self, a, t, shape):
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise):
        return (self._gather(self.sqrt_acp, t, x0.shape) * x0
                + self._gather(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def loss(self, model, batch, cfg):
        x0 = batch['clean'].to(self.device)
        m = batch['mask'].to(self.device)
        cond = build_cond(batch['corrupted'].to(self.device), m, batch['pe'].to(self.device),
                          hole_fill=getattr(cfg, 'hole_fill', 'zero'))
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(torch.cat([xt, cond], dim=1), t)

        target = noise if cfg.predict == 'noise' else x0
        err = (pred - target).abs()
        # whole-patch noise-prediction loss (Palette) with extra weight in the hole.
        glob = err.mean()
        denom = (m.sum() * err.shape[1]).clamp(min=1.0)
        masked = (err * m).sum() / denom
        return (1 - cfg.mask_weight) * glob + cfg.mask_weight * masked

    def predict_x0(self, model, xt, cond, t, clip=None):
        pred = model(torch.cat([xt, cond], dim=1), t)
        sqrt_acp = self._gather(self.sqrt_acp, t, xt.shape)
        sqrt_omacp = self._gather(self.sqrt_one_minus_acp, t, xt.shape)
        x0 = (xt - sqrt_omacp * pred) / sqrt_acp
        if clip is not None:
            x0 = x0.clamp(*clip)
            pred = (xt - sqrt_acp * x0) / sqrt_omacp
        return x0, pred

    @torch.no_grad()
    def sample(self, model, cond, x0_known, mask, predict='noise', clip=(-2.0, 4.0),
               U=1, eta=0.0):
        # DDIM sampling. eta=0 -> deterministic (best point estimate, low MAE);
        # eta=1 -> ancestral DDPM (stochastic). For scientific reconstruction use eta=0.
        device = self.device
        shape = x0_known.shape
        x = torch.randn(shape, device=device)
        keep = (mask == 0).float()
        hole = 1.0 - keep
        for i in reversed(range(self.T)):
            for u in range(U):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                if i > 0:
                    x_known = self.q_sample(x0_known, t, torch.randn_like(x))
                else:
                    x_known = x0_known
                x_in = keep * x_known + hole * x
                if predict == 'noise':
                    x0_pred, eps = self.predict_x0(model, x_in, cond, t, clip=clip)
                else:
                    x0_pred = model(torch.cat([x_in, cond], dim=1), t)
                    if clip is not None:
                        x0_pred = x0_pred.clamp(*clip)
                    sqrt_acp = self._gather(self.sqrt_acp, t, shape)
                    sqrt_omacp = self._gather(self.sqrt_one_minus_acp, t, shape)
                    eps = (x_in - sqrt_acp * x0_pred) / sqrt_omacp

                if i > 0:
                    acp_prev = self._gather(self.acp_prev, t, shape)
                    acp = self._gather(self.acp, t, shape)
                    sigma = eta * torch.sqrt((1 - acp_prev) / (1 - acp) * (1 - acp / acp_prev))
                    dir_xt = torch.sqrt((1 - acp_prev - sigma ** 2).clamp(min=0.0)) * eps
                    noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
                    x_unknown = torch.sqrt(acp_prev) * x0_pred + dir_xt + noise
                else:
                    x_unknown = x0_pred
                x = keep * x_known + hole * x_unknown
                if u < U - 1 and i > 0:
                    beta = self.betas[i]
                    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * torch.randn_like(x)
        return x
