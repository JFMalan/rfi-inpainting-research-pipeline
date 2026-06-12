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

        # Palette input contract: KNOWN region = clean x0, HOLE = the noised state.
        # This prevents the model from copying the answer out of the hole of xt and
        # forces it to inpaint the hole from the clean context + conditioning. The
        # loss is computed ONLY in the hole (the known region is given, so grading it
        # teaches nothing and is overwritten at sampling).
        keep = 1.0 - m
        x_in = keep * x0 + m * xt
        pred = model(torch.cat([x_in, cond], dim=1), t)

        target = noise if cfg.predict == 'noise' else x0
        err = (pred - target).abs()
        denom = (m.sum() * err.shape[1]).clamp(min=1.0)
        return (err * m).sum() / denom

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
    def sample(self, model, cond, x0_known, mask, predict='noise', clip=(-2.0, 4.0), eta=0.0):
        # DDIM sampling matching the training contract: KNOWN region = clean x0_known
        # (never noised), HOLE = the running iterate (never the truth, never fresh
        # noise mid-trajectory). eta=0 -> deterministic point estimate.
        device = self.device
        shape = x0_known.shape
        keep = (mask == 0).float()
        hole = 1.0 - keep
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_in = keep * x0_known + hole * x
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
                sigma = eta * torch.sqrt(((1 - acp_prev) / (1 - acp)) * (1 - acp / acp_prev))
                dir_xt = torch.sqrt((1 - acp_prev - sigma ** 2).clamp(min=0.0)) * eps
                noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
                x = torch.sqrt(acp_prev) * x0_pred + dir_xt + noise
            else:
                x = x0_pred
        return keep * x0_known + hole * x
