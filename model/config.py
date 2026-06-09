from dataclasses import dataclass, field


@dataclass
class Config:
    phase: int = 1
    data_glob: str = ''
    out_dir: str = ''

    pe_channels: int = 4
    base: int = 64
    ch_mult: tuple = (1, 2, 4, 8)
    attn_res: tuple = (32, 16)
    num_res: int = 2
    img_size: int = 256

    timesteps: int = 1000
    predict: str = 'noise'          # 'noise' or 'x0'
    loss_region: str = 'mask'       # 'mask' or 'full'
    mask_weight: float = 0.9        # alpha: mask-region L1 vs weak global term

    # phase 2 mixed masking (inactive in phase 1)
    fake_mask: bool = False
    fake_mask_frac: tuple = (0.05, 0.25)

    batch_size: int = 32
    lr: float = 1e-4
    epochs: int = 40
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    num_workers: int = 6
    augment: bool = True
    seed: int = 0

    sample_every: int = 2
    ckpt_every: int = 2
    max_patches: int = None

    @property
    def in_channels(self):
        return 1 + 1 + 1 + self.pe_channels  # noisy x_t + corrupted + mask + PE


def phase1(**overrides):
    cfg = Config(phase=1, fake_mask=False, loss_region='mask')
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def phase2(**overrides):
    cfg = Config(phase=2, fake_mask=True, loss_region='mask')
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
