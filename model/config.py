from dataclasses import dataclass, field


@dataclass
class Config:
    phase: int = 1
    data_glob: str = ''
    out_dir: str = ''

    target_channels: int = 3        # amplitude + cos(phase) + sin(phase)
    pe_channels: int = 4
    base: int = 64
    ch_mult: tuple = (1, 2, 4, 8, 8)
    attn_res: tuple = (64, 32)
    num_res: int = 2
    img_size: int = 512

    timesteps: int = 1000
    predict: str = 'x0'             # 'noise' or 'x0'; x0 validated leak-free
    mask_weight: float = 0.6        # unused: loss() is now hole-only (Palette contract)
    hole_fill: str = 'mean'         # conditioning hole fill: 'zero'|'mean'|'noise'|'center'

    # phase 2 mixed masking (inactive in phase 1)
    fake_mask: bool = False
    fake_mask_frac: tuple = (0.05, 0.25)
    fake_mask_mode: str = '2d'      # '2d' blobs (force cross-time+freq recovery) | 'bands' (legacy stripes)

    batch_size: int = 32
    lr: float = 2e-4
    epochs: int = 40
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    num_workers: int = 6
    augment: bool = True
    seed: int = 0

    sample_every: int = 2
    ckpt_every: int = 2
    max_patches: int = None

    rand_mask: bool = False        # fresh random training masks (paper Method 2 / anti-memorisation)
    time_roll: bool = False        # random time-axis roll augmentation
    dropout: float = 0.0           # U-Net dropout
    smooth_target: bool = False    # decompose-then-inpaint: predict recoverable smooth bandpass, not noisy amp
    smooth_sigma: float = 1.0      # 2D Gaussian low-pass cutoff; sigma 1.0 cleanly splits
                                   # recoverable structure (smooth ac~0.92) from white noise
                                   # (grain ac~0.01) on real MeerKAT data (sigma sweep 2026-06-21)

    val_eval_patches: int = 64
    val_eval_steps: int = 200  # sampling steps inside val_eval; 200 costs hours per run
    early_stop: bool = True
    patience: int = 8          # consecutive evals with no real complex-MAE gain before stopping
    min_delta: float = 0.002   # complex-MAE units; smaller gains count as no improvement
    min_epochs: int = 20       # never stop before this many epochs

    @property
    def in_channels(self):
        # noisy x_t (target_channels) + masked cond (target_channels) + mask (1) + PE
        return 2 * self.target_channels + 1 + self.pe_channels


def phase1(**overrides):
    cfg = Config(phase=1, fake_mask=False, mask_weight=0.6)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def phase2(**overrides):
    cfg = Config(phase=2, fake_mask=True)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
