from collections import defaultdict

import numpy as np
from scipy.signal.windows import dpss


def dpss_basis(n_chan, hw_ratio, max_modes=None):
    # DPSS modes band-limited to a delay half-width = hw_ratio x Nyquist delay.
    # The field-standard linear gap-filler basis (Ewall-Wice/DAYENU lineage).
    nw = max(1.0, n_chan * hw_ratio / 2.0)
    k = int(np.ceil(2 * nw))
    if max_modes:
        k = min(k, max_modes)
    return dpss(n_chan, nw, k).astype(np.float64)   # (K, n_chan)


def dpss_fill(V, gap, A, lam=0.1):
    # Ridge-regularised DPSS least-squares per row, gaps filled from the fit.
    # V (nrows, N) complex, gap (nrows, N) bool (True = missing). Rows are grouped by
    # gap pattern so the (regularised) operator is built once per pattern.
    out = V.copy()
    reg = lam * np.eye(A.shape[0])
    groups = defaultdict(list)
    packed = np.packbits(gap, axis=1)
    for r in range(V.shape[0]):
        groups[packed[r].tobytes()].append(r)
    for rows in groups.values():
        g = gap[rows[0]]
        ng = int(g.sum())
        if ng == 0 or (~g).sum() < A.shape[0]:   # nothing to fill, or too few good chans to fit
            continue
        good = ~g
        At = A[:, good]                               # (K, ngood)
        M = np.linalg.solve(At @ At.T + reg, At)      # (K, ngood)
        idx = np.array(rows)
        coef = M @ V[idx][:, good].T                  # (K, nrows)
        fill = (A.T @ coef).T                         # (nrows, N)
        out[idx[:, None], np.where(g)[0][None, :]] = fill[:, g]
    return out


def gpr_kernel(n_chan, ell):
    i = np.arange(n_chan)
    d = i[:, None] - i[None, :]
    return np.exp(-0.5 * (d / ell) ** 2)           # sigma_f=1; posterior mean is scale-free in it


def gpr_fill(V, gap, ell=30.0, sigma_n=0.05):
    # GP regression gap-fill along frequency (SE kernel, constant mean). Pagano 2023
    # (arXiv:2210.14927) finds GPR beats DPSS for wide gaps; the soft kernel cutoff, unlike
    # DPSS's hard delay band-limit, lets it place some higher-delay power. Constant-mean GP
    # (subtract per-row observed mean, fill residual, add back) so wide gaps revert to the
    # data level, not zero. Grouped by gap pattern like DPSS.
    out = V.copy()
    K = gpr_kernel(V.shape[1], ell)
    groups = defaultdict(list)
    packed = np.packbits(gap, axis=1)
    for r in range(V.shape[0]):
        groups[packed[r].tobytes()].append(r)
    for rows in groups.values():
        g = gap[rows[0]]
        if not g.any() or not (~g).any():
            continue
        good = ~g
        Koo = K[np.ix_(good, good)] + (sigma_n ** 2) * np.eye(int(good.sum()))
        Kmo = K[np.ix_(g, good)]
        idx = np.array(rows)
        yo = V[idx][:, good]                              # (nrows, ngood) complex
        mu = yo.mean(axis=1, keepdims=True)
        alpha = np.linalg.solve(Koo, (yo - mu).T)         # (ngood, nrows)
        fill = (Kmo @ alpha).T + mu                       # (nrows, nmiss)
        out[idx[:, None], np.where(g)[0][None, :]] = fill
    return out
