import glob
import os

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


def smooth_component(amp, mask, sigma=2.0):
    # recoverable structure: hole-fill along freq, then a 2D Gaussian low-pass. The 2D
    # kernel keeps DIAGONAL fringe structure (a 1D freq filter smears diagonals into the
    # residual). The decorrelated residual (amp - smooth) is irreducible noise and is NOT a
    # training target (decompose-then-inpaint: predict this, resample the residual).
    filled = amp.copy()
    nf = amp.shape[1]
    idx = np.arange(nf)
    for t in range(amp.shape[0]):
        row = filled[t]; keep = mask[t] < 0.5
        if keep.sum() < 4:
            filled[t] = row.mean() if keep.any() else 1.0
            continue
        filled[t] = np.interp(idx, idx[keep], row[keep])
    return gaussian_filter(filled, sigma=sigma, mode='nearest').astype(np.float32)


def positional_encoding(patch_fmin, patch_fmax, band_min, band_max, n_freq, n_time, n_channels):
    f = np.linspace(patch_fmin, patch_fmax, n_freq)
    f_norm = (f - band_min) / (band_max - band_min + 1e-8)
    pe = np.empty((n_channels, n_freq), dtype=np.float32)
    for c in range(n_channels):
        pe[c] = np.sin(f_norm * np.pi / (2.0 ** ((c + 1) / n_channels)))
    pe = np.repeat(pe[:, :, None], n_time, axis=2)  # (C, F, T)
    return np.transpose(pe, (0, 2, 1))               # (C, T, F)


def random_mask(n_time, n_freq, frac_range=(0.05, 0.25)):
    # synthetic RFI-like mask: a few full-width frequency bands + time bursts
    m = np.zeros((n_time, n_freq), dtype=np.float32)
    target = np.random.uniform(*frac_range)
    while m.mean() < target:
        if np.random.rand() < 0.6:  # narrowband: full-time frequency stripe
            w = np.random.randint(1, 6); f0 = np.random.randint(0, n_freq - w)
            m[:, f0:f0 + w] = 1.0
        else:                       # broadband burst: full-freq time stripe
            w = np.random.randint(1, 6); t0 = np.random.randint(0, n_time - w)
            m[t0:t0 + w, :] = 1.0
    return m


class PatchDataset(Dataset):
    def __init__(self, paths, pe_channels=4, augment=False, max_patches=None,
                 split='train', val_frac=0.05, test_frac=0.05, split_seed=1234,
                 amp_only=False, rand_mask=False, time_roll=False, smooth_target=False,
                 smooth_sigma=2.0, clean_target=False, raw_amp=False):
        if smooth_target and clean_target:
            raise ValueError('smooth_target and clean_target are mutually exclusive')
        if isinstance(paths, str):
            paths = [paths]
        self.files = []
        full = []
        bls = []
        for p in paths:
            for fp in sorted(glob.glob(p)):
                with h5py.File(fp, 'r') as f:
                    n = f['clean'].shape[0]
                    self.n_time = int(f.attrs['n_time'])
                    self.n_freq = int(f.attrs['n_freq'])
                    self.band_min = float(f.attrs['freq_min_mhz'])
                    self.band_max = float(f.attrs['freq_max_mhz'])
                    bl_id = f['baseline_id'][:] if 'baseline_id' in f else np.arange(n)
                    if clean_target and 'amp_target' not in f:
                        raise RuntimeError(f'clean_target requested but {fp} has no '
                                           f'amp_target/phase_target fields (re-extract '
                                           f'with CLEAN_DATA in the MS)')
                self.files.append(fp)
                fidx = len(self.files) - 1
                for i in range(n):
                    full.append((fidx, i))
                    bls.append((fidx, int(bl_id[i])))
        if not full:
            raise RuntimeError(f"no patches found in {paths}")

        # split by (file, baseline) so a baseline's freq tiles never straddle train/val/test
        groups = {}
        for k, key in enumerate(bls):
            groups.setdefault(key, []).append(k)
        gkeys = list(groups.keys())
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(gkeys))
        n_val = int(len(gkeys) * val_frac)
        n_test = int(len(gkeys) * test_frac)
        sel = {'train': perm[n_test + n_val:], 'val': perm[n_test:n_test + n_val],
               'test': perm[:n_test]}[split]
        chosen = sorted(int(k) for gi in sel for k in groups[gkeys[gi]])

        self.index = [full[k] for k in chosen]
        if split == 'train' and max_patches is not None and max_patches < len(self.index):
            sub = rng.choice(len(self.index), size=max_patches, replace=False)
            self.index = [self.index[k] for k in sorted(sub)]

        self.split = split
        self.augment = augment and split == 'train'
        self.amp_only = amp_only
        self.rand_mask = rand_mask and split == 'train'
        self.time_roll = time_roll and split == 'train'
        self.smooth_target = smooth_target
        self.smooth_sigma = smooth_sigma
        self.clean_target = clean_target
        self.raw_amp = raw_amp
        self.pe_channels = pe_channels
        self._handles = {}
        self._pe_cache = {}

    def _file(self, fidx):
        h = self._handles.get(fidx)
        if h is None:
            h = h5py.File(self.files[fidx], 'r')
            self._handles[fidx] = h
        return h

    def _pe(self, fmin, fmax):
        key = (round(float(fmin), 3), round(float(fmax), 3))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = positional_encoding(fmin, fmax, self.band_min, self.band_max,
                                     self.n_freq, self.n_time, self.pe_channels)
            self._pe_cache[key] = pe
        return pe

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fidx, row = self.index[i]
        f = self._file(fidx)
        clean = f['clean'][row].astype(np.float32)
        corrupted = f['corrupted'][row].astype(np.float32)
        mask = f['mask'][row].astype(np.float32)
        phase = f['phase'][row].astype(np.float32)
        if self.clean_target:
            amp_t = f['amp_target'][row].astype(np.float32)
            phase_t = f['phase_target'][row].astype(np.float32)

        if self.raw_amp:
            # undo the divisive norm (Massoud R0 rung: no div-norm in the recipe)
            div = f['dn_divisor'][row].astype(np.float32)
            clean = clean * div
            corrupted = corrupted * div
            if self.clean_target:
                amp_t = amp_t * div

        if 'freq_min_patch' in f:
            fmin = float(f['freq_min_patch'][row])
            fmax = float(f['freq_max_patch'][row])
        else:
            fmin = float(f.attrs['freq_min_mhz'])
            fmax = float(f.attrs['freq_max_mhz'])

        if self.augment and np.random.rand() < 0.5:
            clean = clean[::-1].copy()
            corrupted = corrupted[::-1].copy()
            mask = mask[::-1].copy()
            phase = phase[::-1].copy()
            if self.clean_target:
                amp_t = amp_t[::-1].copy()
                phase_t = phase_t[::-1].copy()

        if self.time_roll:
            sh = np.random.randint(0, clean.shape[0])
            clean = np.roll(clean, sh, axis=0)
            corrupted = np.roll(corrupted, sh, axis=0)
            mask = np.roll(mask, sh, axis=0)
            phase = np.roll(phase, sh, axis=0)
            if self.clean_target:
                amp_t = np.roll(amp_t, sh, axis=0)
                phase_t = np.roll(phase_t, sh, axis=0)

        if self.rand_mask:
            # fresh random hole each time -> model can't memorise per-patch holes.
            # corrupted==clean (context is the true signal); the hole is hidden by
            # build_cond regardless, so the hole's corrupted value is irrelevant.
            mask = random_mask(clean.shape[0], clean.shape[1])
            corrupted = clean.copy()

        # noise-free target: x0 = pre-noise amplitude AND phase (shared noisy divisor);
        # conditioning stays the noisy observation. Legacy smooth_target keeps the
        # decompose-era behaviour for old checkpoints; plain path targets the noisy amp.
        if self.clean_target:
            target_amp, target_phase = amp_t, phase_t
        elif self.smooth_target:
            target_amp, target_phase = smooth_component(clean, mask, self.smooth_sigma), phase
        else:
            target_amp, target_phase = clean, phase

        if self.amp_only:
            clean_t = target_amp[None]
            corrupted_t = corrupted[None]
        else:
            # 3-channel: amplitude + cos(phase) + sin(phase).
            clean_t = np.stack([target_amp, np.cos(target_phase), np.sin(target_phase)], axis=0)
            corrupted_t = np.stack([corrupted, np.cos(phase), np.sin(phase)], axis=0)
        pe = self._pe(fmin, fmax)

        return {
            'clean': torch.from_numpy(clean_t),
            'corrupted': torch.from_numpy(corrupted_t),
            'mask': torch.from_numpy(mask)[None],           # (1, T, F)
            'pe': torch.from_numpy(pe.copy()),              # (C, T, F)
            'fmin': torch.tensor(fmin, dtype=torch.float32),
            'fmax': torch.tensor(fmax, dtype=torch.float32),
        }

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles = {}


def fake_mask(real_flags, frac_range=(0.05, 0.25), width_range=(8, 32), max_tries=200,
              mode='2d'):
    # mixed-masking (Massoud): artificial holes over UNFLAGGED pixels, where the observed
    # data is a known self-supervised target. real-flagged pixels carry no target.
    # mode='2d': RECTANGULAR BLOBS bounded in both time and freq. A full-time freq stripe
    # is filled by 1D freq-interp = the smooth/bandpass target, so the model can't beat
    # interp on it (audit 2026-06-21). 2D blobs force the model to use cross-time AND
    # cross-freq context, which is the only setting where it can beat interpolation.
    # mode='bands': legacy full-width stripes (kept for comparison).
    # mode='mixed': blobs + full-time freq stripes + broadband time bursts. The real flags
    # are dominated by persistent full-time freq bands, so training only on 2D blobs leaves
    # the model extrapolating on the exact geometry it deploys to; the mix keeps the blob
    # case (where it beats interp) AND covers the band geometry (where it must at least match
    # interp, which still helps the image vs flagging). Stripes are widened toward the
    # persistent-band scale.
    n_time, n_freq = real_flags.shape
    fm = np.zeros((n_time, n_freq), dtype=np.float32)
    target = np.random.uniform(*frac_range)
    unflagged = real_flags < 0.5
    avail = unflagged.mean()
    target = min(target, max(avail * 0.8, 1e-3))
    wlo, whi = width_range
    band_hi = max(whi, n_freq // 8)
    tries = 0
    while (fm * unflagged).mean() < target and tries < max_tries:
        tries += 1
        if mode == 'mixed':
            r = np.random.rand()
            shape = 'blob' if r < 0.35 else ('stripe' if r < 0.8 else 'burst')
        elif mode == '2d':
            shape = 'blob'
        elif mode == 'stripe':
            shape = 'stripe'
        else:
            shape = 'stripe' if np.random.rand() < 0.6 else 'burst'
        if shape == 'blob':
            wf = np.random.randint(wlo, whi + 1); wt = np.random.randint(wlo, whi + 1)
            f0 = np.random.randint(0, max(1, n_freq - wf)); t0 = np.random.randint(0, max(1, n_time - wt))
            fm[t0:t0 + wt, f0:f0 + wf] = 1.0
        elif shape == 'stripe':
            hi = band_hi if mode in ('mixed', 'stripe') else whi
            w = np.random.randint(wlo, hi + 1); f0 = np.random.randint(0, max(1, n_freq - w))
            fm[:, f0:f0 + w] = 1.0
        else:
            w = np.random.randint(max(2, wlo // 2), whi // 2 + 1); t0 = np.random.randint(0, max(1, n_time - w))
            fm[t0:t0 + w, :] = 1.0
    fm = fm * unflagged
    return fm


class RealDataset(Dataset):
    def __init__(self, paths, pe_channels=4, augment=False, max_patches=None,
                 split='train', val_frac=0.05, test_frac=0.05, split_seed=1234,
                 fake_mask_frac=(0.05, 0.25), smooth_target=False, smooth_sigma=2.0,
                 fake_mask_mode='2d'):
        if isinstance(paths, str):
            paths = [paths]
        self.files = []
        full = []
        bls = []
        stored_split = []   # per-sample 0/1 test flag if the file carries one
        have_split = True
        for p in paths:
            for fp in sorted(glob.glob(p)):
                with h5py.File(fp, 'r') as f:
                    n = f['data'].shape[0]
                    self.n_time = int(f.attrs['n_time'])
                    self.n_freq = int(f.attrs['n_freq'])
                    self.band_min = float(f.attrs['freq_min_mhz'])
                    self.band_max = float(f.attrs['freq_max_mhz'])
                    sp = f['split'][:] if 'split' in f else None
                    bl_id = f['baseline_id'][:] if 'baseline_id' in f else np.arange(n)
                self.files.append(fp)
                fidx = len(self.files) - 1
                for i in range(n):
                    full.append((fidx, i))
                    bls.append((fidx, int(bl_id[i])))
                if sp is None:
                    have_split = False
                else:
                    stored_split.extend(int(v) for v in sp)
        if not full:
            raise RuntimeError(f"no baselines found in {paths}")

        # group sample indices by (file, baseline) so a baseline's freq tiles stay together
        def by_baseline(ids):
            groups = {}
            for k in ids:
                groups.setdefault(bls[k], []).append(k)
            return list(groups.values())

        rng = np.random.default_rng(split_seed)
        if have_split:
            # extractor reserved test baselines; val carved from the train pool by baseline
            test_ids = [k for k, s in enumerate(stored_split) if s == 1]
            pool_groups = by_baseline([k for k, s in enumerate(stored_split) if s == 0])
            perm = rng.permutation(len(pool_groups))
            n_val = int(len(pool_groups) * val_frac)
            val_ids = [k for gi in perm[:n_val] for k in pool_groups[gi]]
            train_ids = [k for gi in perm[n_val:] for k in pool_groups[gi]]
        else:
            groups = by_baseline(range(len(full)))
            perm = rng.permutation(len(groups))
            n_val = int(len(groups) * val_frac)
            n_test = int(len(groups) * test_frac)
            test_ids = [k for gi in perm[:n_test] for k in groups[gi]]
            val_ids = [k for gi in perm[n_test:n_test + n_val] for k in groups[gi]]
            train_ids = [k for gi in perm[n_test + n_val:] for k in groups[gi]]
        chosen = {'train': train_ids, 'val': val_ids, 'test': test_ids}[split]

        self.index = [full[k] for k in sorted(chosen)]
        if split == 'train' and max_patches is not None and max_patches < len(self.index):
            sub = rng.choice(len(self.index), size=max_patches, replace=False)
            self.index = [self.index[k] for k in sorted(sub)]

        self.split = split
        self.augment = augment and split == 'train'
        self.fake_mask_frac = fake_mask_frac
        self.fake_mask_mode = fake_mask_mode
        self.smooth_target = smooth_target
        self.smooth_sigma = smooth_sigma
        self.pe_channels = pe_channels
        self._handles = {}
        self._pe_cache = {}

    def _file(self, fidx):
        h = self._handles.get(fidx)
        if h is None:
            h = h5py.File(self.files[fidx], 'r')
            self._handles[fidx] = h
        return h

    def __len__(self):
        return len(self.index)

    def _pe_band(self, fmin, fmax):
        key = (round(fmin, 3), round(fmax, 3))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = positional_encoding(fmin, fmax, self.band_min, self.band_max,
                                     self.n_freq, self.n_time, self.pe_channels)
            self._pe_cache[key] = pe
        return pe

    def __getitem__(self, i):
        fidx, row = self.index[i]
        f = self._file(fidx)
        data = f['data'][row].astype(np.float32)
        phase = f['phase'][row].astype(np.float32)
        real_flags = f['flags'][row].astype(np.float32)
        divisor = f['dn_divisor'][row].astype(np.float32) if 'dn_divisor' in f \
            else np.ones_like(data)

        if 'freq_min_patch' in f:
            fmin = float(f['freq_min_patch'][row]); fmax = float(f['freq_max_patch'][row])
        else:
            fmin, fmax = self.band_min, self.band_max

        if self.augment and np.random.rand() < 0.5:
            data = data[::-1].copy()
            phase = phase[::-1].copy()
            real_flags = real_flags[::-1].copy()
            divisor = divisor[::-1].copy()

        # val holes are fixed per sample so val metrics are comparable across epochs and runs
        if self.split != 'train':
            rs = np.random.get_state()
            np.random.seed(1000003 + i)
        fm = fake_mask(real_flags, self.fake_mask_frac, mode=self.fake_mask_mode)
        if self.split != 'train':
            np.random.set_state(rs)

        # decompose-then-inpaint: the self-sup target becomes the recoverable smooth
        # amplitude (real_flags masked out so flagged junk doesn't pollute the smoothing).
        # context conditioning still hides flags+fake holes; only the target changes.
        data_t = smooth_component(data, real_flags, self.smooth_sigma) if self.smooth_target else data

        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        obs = np.stack([data_t, cos_p, sin_p], axis=0)
        hidden = np.clip(real_flags + fm, 0.0, 1.0)        # conditioning hides both
        pe = self._pe_band(fmin, fmax)

        return {
            'obs': torch.from_numpy(obs),                  # observed = self-sup target
            'real_flags': torch.from_numpy(real_flags)[None],
            'fake_mask': torch.from_numpy(fm)[None],       # loss region
            'hidden': torch.from_numpy(hidden)[None],      # conditioning hole
            'divisor': torch.from_numpy(divisor),
            'pe': torch.from_numpy(pe.copy()),
            'fmin': torch.tensor(self.band_min, dtype=torch.float32),
            'fmax': torch.tensor(self.band_max, dtype=torch.float32),
        }

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles = {}


def build_cond(corrupted, mask, pe, hole_fill='zero', chan_means=None):
    # masked (RFI) pixels are hidden from the network; it inpaints from context.
    # the hole is filled with an in-distribution value (the per-channel mean) so it
    # is not an out-of-distribution cliff on non-zero-mean data.
    keep = 1.0 - mask
    if hole_fill == 'zero':
        known = corrupted * keep
    elif hole_fill == 'mean':
        if chan_means is None:
            chan_means = corrupted.new_tensor([1.0, 0.0, 0.0][:corrupted.shape[1]])
        fill = chan_means.view(1, -1, 1, 1)
        known = corrupted * keep + fill * mask
    elif hole_fill == 'noise':
        known = corrupted * keep + torch.randn_like(corrupted) * mask
    elif hole_fill == 'center':
        if chan_means is None:
            chan_means = corrupted.new_tensor([1.0, 0.0, 0.0][:corrupted.shape[1]])
        c = chan_means.view(1, -1, 1, 1)
        known = (corrupted - c) * keep
    else:
        raise ValueError(hole_fill)
    return torch.cat([known, mask, pe], dim=1)
