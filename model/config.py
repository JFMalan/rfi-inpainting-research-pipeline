from dataclasses import dataclass, field


@dataclass
class Config:
    phase: int = 1
    data_glob: str = ''
    out_dir: str = ''

    target_channels: int = 3        # amplitude + cos(phase) + sin(phase)
    pe_channels: int = 4
    base: int = 64
    ch_mult: tuple = (1, 2, 4, 8)
    attn_res: tuple = (32, 16)
    num_res: int = 2
    img_size: int = 256

    timesteps: int = 1000
    predict: str = 'noise'          # 'noise' or 'x0'
    # the masked region of xt is replaced with noise in loss(), so the model must
    # inpaint the hole from context. supervise mostly in the hole (we have true x0
    # in the supervised phase) + a weak whole-patch term for the known region.
    mask_weight: float = 0.8

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

    val_eval_patches: int = 64
    early_stop: bool = True
    patience: int = 8          # consecutive evals with no real PSNR gain before stopping
    min_delta: float = 0.05    # dB; smaller gains count as no improvement (sampler is noisy)
    min_epochs: int = 20       # never stop before this many epochs

    @property
    def in_channels(self):
        # noisy x_t (target_channels) + masked cond (target_channels) + mask (1) + PE
        return 2 * self.target_channels + 1 + self.pe_channels


def phase1(**overrides):
    cfg = Config(phase=1, fake_mask=False, mask_weight=0.8)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def phase2(**overrides):
    cfg = Config(phase=2, fake_mask=True)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
