import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def positional_encoding(freq_min, freq_max, n_freq, n_time, n_channels):
    f = np.linspace(freq_min, freq_max, n_freq)
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-8)
    pe = np.empty((n_channels, n_freq), dtype=np.float32)
    for c in range(n_channels):
        pe[c] = np.sin(f_norm * np.pi / (2.0 ** ((c + 1) / n_channels)))
    pe = np.repeat(pe[:, :, None], n_time, axis=2)  # (C, F, T)
    return np.transpose(pe, (0, 2, 1))               # (C, T, F)


class PatchDataset(Dataset):
    def __init__(self, paths, pe_channels=4, augment=False, max_patches=None,
                 split='train', val_frac=0.05, test_frac=0.05, split_seed=1234):
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
            pe = positional_encoding(fmin, fmax, self.n_freq, self.n_time, self.pe_channels)
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

        if 'freq_min_patch' in f:
            fmin = float(f['freq_min_patch'][row])
            fmax = float(f['freq_max_patch'][row])
        else:
            fmin = float(f.attrs['freq_min_mhz'])
            fmax = float(f.attrs['freq_max_mhz'])

        if self.augment:
            if np.random.rand() < 0.5:
                clean = clean[::-1].copy()
                corrupted = corrupted[::-1].copy()
                mask = mask[::-1].copy()

        pe = self._pe(fmin, fmax)

        return {
            'clean': torch.from_numpy(clean)[None],         # (1, T, F)
            'corrupted': torch.from_numpy(corrupted)[None],
            'mask': torch.from_numpy(mask)[None],
            'pe': torch.from_numpy(pe.copy()),              # (C, T, F)
        }

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles = {}


def build_cond(corrupted, mask, pe):
    return torch.cat([corrupted, mask, pe], dim=1)
