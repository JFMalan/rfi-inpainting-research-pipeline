import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, heads=4, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.heads = heads
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        scale = (c // self.heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', q, k) * scale, dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode='nearest'))


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch=1, base=64, ch_mult=(1, 2, 4, 8),
                 attn_res=(16,), num_res=2, img_size=256, groups=8):
        super().__init__()
        t_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.time_dim = base

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        chs = [base]
        ch = base
        res = img_size
        self.downs = nn.ModuleList()
        for level, mult in enumerate(ch_mult):
            out = base * mult
            for _ in range(num_res):
                blocks = nn.ModuleList([ResBlock(ch, out, t_dim, groups)])
                ch = out
                if res in attn_res:
                    blocks.append(AttnBlock(ch, groups=groups))
                self.downs.append(blocks)
                chs.append(ch)
            if level != len(ch_mult) - 1:
                self.downs.append(nn.ModuleList([Down(ch)]))
                chs.append(ch)
                res //= 2

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, t_dim, groups),
            AttnBlock(ch, groups=groups),
            ResBlock(ch, ch, t_dim, groups),
        ])

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out = base * mult
            for _ in range(num_res + 1):
                blocks = nn.ModuleList([ResBlock(ch + chs.pop(), out, t_dim, groups)])
                ch = out
                if res in attn_res:
                    blocks.append(AttnBlock(ch, groups=groups))
                self.ups.append(blocks)
            if level != 0:
                self.ups.append(nn.ModuleList([Up(ch)]))
                res *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(groups, ch), nn.SiLU(), nn.Conv2d(ch, out_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t = self.time_mlp(timestep_embedding(t, self.time_dim))
        h = self.in_conv(x)
        skips = [h]
        for blocks in self.downs:
            if isinstance(blocks[0], Down):
                h = blocks[0](h)
            else:
                for b in blocks:
                    h = b(h, t) if isinstance(b, ResBlock) else b(h)
            skips.append(h)
        for b in self.mid:
            h = b(h, t) if isinstance(b, ResBlock) else b(h)
        for blocks in self.ups:
            if isinstance(blocks[0], Up):
                h = blocks[0](h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                for b in blocks:
                    h = b(h, t) if isinstance(b, ResBlock) else b(h)
        return self.out(h)
