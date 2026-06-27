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
