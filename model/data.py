import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def positional_encoding(patch_fmin, patch_fmax, band_min, band_max, n_freq, n_time, n_channels):
    f = np.linspace(patch_fmin, patch_fmax, n_freq)
    f_norm = (f - band_min) / (band_max - band_min + 1e-8)
    pe = np.empty((n_channels, n_freq), dtype=np.float32)
    for c in range(n_channels):
        pe[c] = np.sin(f_norm * np.pi / (2.0 ** ((c + 1) / n_channels)))
    pe = np.repeat(pe[:, :, None], n_time, axis=2)  # (C, F, T)
    return np.transpose(pe, (0, 2, 1))               # (C, T, F)


class PatchDataset(Dataset):
    def __init__(self, paths, pe_channels=4, augment=False, max_patches=None,
                 split='train', val_frac=0.05, test_frac=0.05, split_seed=1234,
                 amp_only=False):
        if isinstance(paths, str):
            paths = [paths]
        self.files = []
        full = []
        for p in paths:
            for fp in sorted(glob.glob(p)):
                with h5py.File(fp, 'r') as f:
                    n = f['clean'].shape[0]
                    self.n_time = int(f.attrs['n_time'])
                    self.n_freq = int(f.attrs['n_freq'])
                    self.band_min = float(f.attrs['freq_min_mhz'])
                    self.band_max = float(f.attrs['freq_max_mhz'])
                self.files.append(fp)
                fidx = len(self.files) - 1
                for i in range(n):
                    full.append((fidx, i))
        if not full:
            raise RuntimeError(f"no patches found in {paths}")

        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(full))
        n_val = int(len(full) * val_frac)
        n_test = int(len(full) * test_frac)
        test_ids = perm[:n_test]
        val_ids = perm[n_test:n_test + n_val]
        train_ids = perm[n_test + n_val:]
        chosen = {'train': train_ids, 'val': val_ids, 'test': test_ids}[split]

        self.index = [full[k] for k in sorted(chosen)]
        if split == 'train' and max_patches is not None and max_patches < len(self.index):
            sub = rng.choice(len(self.index), size=max_patches, replace=False)
            self.index = [self.index[k] for k in sorted(sub)]

        self.split = split
        self.augment = augment and split == 'train'
        self.amp_only = amp_only
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

        if self.amp_only:
            clean_t = clean[None]
            corrupted_t = corrupted[None]
        else:
            cos_p = np.cos(phase)
            sin_p = np.sin(phase)
            # 3-channel: amplitude + cos(phase) + sin(phase).
            clean_t = np.stack([clean, cos_p, sin_p], axis=0)
            corrupted_t = np.stack([corrupted, cos_p, sin_p], axis=0)
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
