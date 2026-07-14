import argparse

import h5py
import numpy as np
from scipy.ndimage import uniform_filter


def masked_corr(x, m, lag, axis):
    a = x.take(range(0, x.shape[axis] - lag), axis=axis)
    b = x.take(range(lag, x.shape[axis]), axis=axis)
    ka = m.take(range(0, m.shape[axis] - lag), axis=axis)
    kb = m.take(range(lag, m.shape[axis]), axis=axis)
    ok = (ka > 0.5) & (kb > 0.5)
    if ok.sum() < 200:
        return np.nan
    a, b = a[ok], b[ok]
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).mean() * (b * b).mean())
    return float((a * b).mean() / d) if d > 1e-12 else np.nan


def tile_stats(amp, phase, flags, rfi_amp=None):
    trust = flags < 0.5
    if trust.sum() < 500:
        return None
    hp = amp - uniform_filter(amp, 5, mode='nearest')
    clear = uniform_filter(flags, 5, mode='nearest') < 1e-6
    if clear.sum() < 100:
        clear = trust
    s = {
        'grain': float(hp[clear].std()),
        'amp_p50': float(np.median(amp[trust])),
        'amp_iqr': float(np.subtract(*np.percentile(amp[trust], [75, 25]))),
        'flag_frac': float(flags.mean()),
        'ac_freq8': masked_corr(amp, trust.astype(np.float32), 8, axis=1),
        'ac_time8': masked_corr(amp, trust.astype(np.float32), 8, axis=0),
    }
    dp = np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :-1])))
    ok = trust[:, 1:] & trust[:, :-1]
    s['dphase'] = float(np.abs(dp[ok]).mean()) if ok.sum() > 200 else np.nan
    if rfi_amp is not None and (~trust).sum() > 200:
        s['rfi_contrast'] = float(np.median(rfi_amp[~trust]) / max(np.median(amp[trust]), 1e-9))
    return s


KEYS = ['grain', 'amp_p50', 'amp_iqr', 'flag_frac', 'ac_freq8', 'ac_time8',
        'dphase', 'rfi_contrast']
DESC = {
    'grain':        '5x5 high-pass std, unflagged (noise regime)',
    'amp_p50':      'median normalized amp, unflagged',
    'amp_iqr':      'amp IQR, unflagged (structure depth)',
    'flag_frac':    'flagged fraction per tile',
    'ac_freq8':     'autocorr lag-8 along freq (structure scale)',
    'ac_time8':     'autocorr lag-8 along time (structure scale)',
    'dphase':       'mean |dphase/dchan|, rad (pi/2=noise-dominated)',
    'rfi_contrast': 'median flagged amp / median unflagged amp',
}


def summarise(name, path, kind, cap):
    hf = h5py.File(path, 'r')
    amp_key, flag_key = ('clean', 'mask') if kind == 'sim' else ('data', 'flags')
    n = hf[amp_key].shape[0]
    idx = np.linspace(0, n - 1, min(cap, n)).astype(int)
    acc = {k: [] for k in KEYS}
    for u in idx:
        amp = hf[amp_key][u].astype(np.float32)
        phase = hf['phase'][u].astype(np.float32)
        flags = hf[flag_key][u].astype(np.float32)
        rfi = hf['corrupted'][u].astype(np.float32) if kind == 'sim' else amp
        s = tile_stats(amp, phase, flags, rfi_amp=rfi)
        if s is None:
            continue
        for k, v in s.items():
            if np.isfinite(v):
                acc[k].append(v)
    hf.close()
    out = {}
    for k in KEYS:
        out[k] = np.percentile(acc[k], [10, 50, 90]) if acc[k] else None
    print(f"{name}: {len(acc['grain'])} tiles from {path}", flush=True)
    return out


def main(args):
    real = summarise('real', args.real, 'real', args.cap)
    sim = summarise('sim', args.sim, 'sim', args.cap)
    print()
    print(f"{'statistic':<14} {'real p10/p50/p90':>26} {'sim p10/p50/p90':>26} {'p50 ratio':>10}   note")
    for k in KEYS:
        r, s = real[k], sim[k]
        if r is None or s is None:
            print(f"{k:<14} {'-':>26} {'-':>26} {'-':>10}   {DESC[k]}")
            continue
        rs = '/'.join(f'{v:.3f}' for v in r)
        ss = '/'.join(f'{v:.3f}' for v in s)
        ratio = r[1] / s[1] if abs(s[1]) > 1e-9 else np.inf
        print(f"{k:<14} {rs:>26} {ss:>26} {ratio:>10.2f}   {DESC[k]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--sim', required=True)
    ap.add_argument('--cap', type=int, default=200)
    main(ap.parse_args())
