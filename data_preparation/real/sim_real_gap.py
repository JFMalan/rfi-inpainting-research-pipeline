import argparse
import time
import glob
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance, ks_2samp

t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def load_sim(paths, n):
    files = sorted(f for p in paths for f in glob.glob(p))
    if not files:
        raise RuntimeError(f"no sim files match {paths}")
    amp, mask, phase, attrs = [], [], [], None
    per = max(1, n // len(files))
    for fp in files:
        with h5py.File(fp, 'r') as f:
            if attrs is None:
                attrs = {k: f.attrs[k] for k in f.attrs}
            tot = f['clean'].shape[0]
            k = min(per, tot)
            idx = np.linspace(0, tot - 1, k).astype(int)
            amp.append(f['clean'][idx].astype(np.float32))
            mask.append(f['mask'][idx].astype(np.float32))
            phase.append(f['phase'][idx].astype(np.float32))
        log(f"  sim {Path(fp).parent.name}: took {k}/{tot}")
        if sum(a.shape[0] for a in amp) >= n:
            break
    return (np.concatenate(amp)[:n], np.concatenate(mask)[:n],
            np.concatenate(phase)[:n], attrs)


def load_real(path, n):
    with h5py.File(path, 'r') as f:
        attrs = {k: f.attrs[k] for k in f.attrs}
        tot = f['data'].shape[0]
        k = min(n, tot)
        idx = np.linspace(0, tot - 1, k).astype(int)
        amp = f['data'][idx].astype(np.float32)
        flags = f['flags'][idx].astype(np.float32)
        phase = f['phase'][idx].astype(np.float32)
    log(f"  real: took {k}/{tot}")
    return amp, flags, phase, attrs


def stats(x):
    p = np.percentile(x, [1, 5, 50, 95, 99, 99.9])
    return dict(mean=x.mean(), std=x.std(), p1=p[0], p5=p[1],
               p50=p[2], p95=p[3], p99=p[4], p999=p[5])


def lag1_autocorr(imgs, valid, axis):
    # mean lag-1 Pearson correlation along `axis`, computed per-image over valid pixels
    acc = []
    for im, v in zip(imgs, valid):
        a = im if axis == 1 else im.T
        m = v if axis == 1 else v.T
        x0 = a[:, :-1]; x1 = a[:, 1:]
        good = m[:, :-1] & m[:, 1:]
        if good.sum() < 50:
            continue
        x0 = x0[good]; x1 = x1[good]
        x0 = x0 - x0.mean(); x1 = x1 - x1.mean()
        d = np.sqrt((x0 * x0).sum() * (x1 * x1).sum())
        if d > 0:
            acc.append((x0 * x1).sum() / d)
    return float(np.mean(acc)) if acc else np.nan


def radial_psd(imgs, valid):
    # radially-averaged 2D power spectrum; flagged pixels mean-filled per image so the
    # FFT sees no edge cliff (we compare clean texture, not the holes)
    n = imgs.shape[1]
    cy, cx = n // 2, n // 2
    yy, xx = np.ogrid[:n, :n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rmax = n // 2
    acc = np.zeros(rmax)
    cnt = 0
    for im, v in zip(imgs, valid):
        filled = im.copy()
        if (~v).any():
            filled[~v] = im[v].mean() if v.any() else 0.0
        filled = filled - filled.mean()
        ps = np.abs(np.fft.fftshift(np.fft.fft2(filled))) ** 2
        prof = np.array([ps[r == i].mean() for i in range(rmax)])
        acc += prof
        cnt += 1
    return acc / max(cnt, 1)


def flag_band_profile(masks):
    # fraction of pixels flagged per frequency channel, averaged over samples/time
    return masks.mean(axis=(0, 1))


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    log("loading sim...")
    s_amp, s_mask, s_phase, s_attrs = load_sim(args.sim, args.n)
    log(f"sim loaded: {s_amp.shape}")
    log("loading real...")
    r_amp, r_flags, r_phase, r_attrs = load_real(args.real, args.n)
    log(f"real loaded: {r_amp.shape}")

    fmin = float(r_attrs.get('freq_min_mhz', 900.0))
    fmax = float(r_attrs.get('freq_max_mhz', 1650.0))
    nfreq = r_amp.shape[2]
    freqs = np.linspace(fmin, fmax, nfreq)

    s_valid = s_mask < 0.5
    r_valid = r_flags < 0.5
    log(f"sim flag frac={1 - s_valid.mean():.3f}  real flag frac={1 - r_valid.mean():.3f}")

    # ---------- 1. AMPLITUDE ----------
    log("amplitude distribution...")
    s_clean = s_amp[s_valid]
    r_clean = r_amp[r_valid]
    s_clean = s_clean[np.isfinite(s_clean)]
    r_clean = r_clean[np.isfinite(r_clean)]

    ss = stats(s_clean); rs = stats(r_clean)
    print("\n=== AMPLITUDE (unflagged pixels) ===", flush=True)
    hdr = f"{'':6}" + "".join(f"{k:>9}" for k in ss)
    print(hdr, flush=True)
    print(f"{'sim':6}" + "".join(f"{ss[k]:9.3f}" for k in ss), flush=True)
    print(f"{'real':6}" + "".join(f"{rs[k]:9.3f}" for k in rs), flush=True)

    # subsample for KS (expensive on millions of points)
    rng = np.random.default_rng(0)
    sa = rng.choice(s_clean, min(200000, s_clean.size), replace=False)
    ra = rng.choice(r_clean, min(200000, r_clean.size), replace=False)
    ks = ks_2samp(sa, ra)
    wd = wasserstein_distance(sa, ra)
    print(f"\nKS statistic = {ks.statistic:.4f} (p={ks.pvalue:.2e})", flush=True)
    print(f"Wasserstein distance = {wd:.4f}", flush=True)
    print(f"Wasserstein / real-std = {wd / rs['std']:.3f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    hi = np.percentile(np.concatenate([sa, ra]), 99.5)
    bins = np.linspace(0, hi, 120)
    ax[0].hist(s_clean, bins=bins, density=True, alpha=0.55, label='sim clean')
    ax[0].hist(r_clean, bins=bins, density=True, alpha=0.55, label='real unflagged')
    ax[0].set_xlabel('amplitude (divnorm)'); ax[0].set_ylabel('density')
    ax[0].set_title('amplitude distribution'); ax[0].legend()
    lbins = np.linspace(np.log10(max(sa.min(), 1e-3)),
                        np.log10(np.percentile(np.concatenate([sa, ra]), 99.9)), 120)
    ax[1].hist(np.log10(np.clip(s_clean, 1e-3, None)), bins=lbins, density=True, alpha=0.55, label='sim')
    ax[1].hist(np.log10(np.clip(r_clean, 1e-3, None)), bins=lbins, density=True, alpha=0.55, label='real')
    ax[1].set_xlabel('log10 amplitude'); ax[1].set_ylabel('density')
    ax[1].set_title('amplitude (log scale)'); ax[1].legend()
    plt.tight_layout(); plt.savefig(out / 'amp_dist.png', dpi=120); plt.close()

    # ---------- 2. SPATIAL STRUCTURE ----------
    log("spatial structure (autocorr + PSD)...")
    nstruct = min(150, s_amp.shape[0], r_amp.shape[0])
    s_ac_f = lag1_autocorr(s_amp[:nstruct], s_valid[:nstruct], axis=1)
    s_ac_t = lag1_autocorr(s_amp[:nstruct], s_valid[:nstruct], axis=0)
    r_ac_f = lag1_autocorr(r_amp[:nstruct], r_valid[:nstruct], axis=1)
    r_ac_t = lag1_autocorr(r_amp[:nstruct], r_valid[:nstruct], axis=0)
    print("\n=== LAG-1 AUTOCORRELATION (clean amp) ===", flush=True)
    print(f"          along-freq   along-time", flush=True)
    print(f"sim       {s_ac_f:9.4f}   {s_ac_t:9.4f}", flush=True)
    print(f"real      {r_ac_f:9.4f}   {r_ac_t:9.4f}", flush=True)

    s_psd = radial_psd(s_amp[:nstruct], s_valid[:nstruct])
    r_psd = radial_psd(r_amp[:nstruct], r_valid[:nstruct])
    k = np.arange(1, len(s_psd))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(k, s_psd[1:] / s_psd[1:].sum(), label='sim', lw=1.3)
    ax.loglog(k, r_psd[1:] / r_psd[1:].sum(), label='real', lw=1.3)
    ax.set_xlabel('radial spatial frequency'); ax.set_ylabel('normalised power')
    ax.set_title('radially-averaged 2D PSD (clean amp)'); ax.legend()
    plt.tight_layout(); plt.savefig(out / 'psd.png', dpi=120); plt.close()

    # ---------- 3. PHASE ----------
    log("phase structure...")
    s_ph = s_phase[s_valid]; r_ph = r_phase[r_valid]
    s_ph = s_ph[np.isfinite(s_ph)]; r_ph = r_ph[np.isfinite(r_ph)]
    # phase autocorr via cos: lag-1 of cos(phase) along freq captures coherence
    s_cos = np.cos(s_phase); r_cos = np.cos(r_phase)
    s_pac_f = lag1_autocorr(s_cos[:nstruct], s_valid[:nstruct], axis=1)
    r_pac_f = lag1_autocorr(r_cos[:nstruct], r_valid[:nstruct], axis=1)
    s_pac_t = lag1_autocorr(s_cos[:nstruct], s_valid[:nstruct], axis=0)
    r_pac_t = lag1_autocorr(r_cos[:nstruct], r_valid[:nstruct], axis=0)
    print("\n=== PHASE ===", flush=True)
    print(f"cos(phase) lag-1 autocorr   along-freq   along-time", flush=True)
    print(f"sim                         {s_pac_f:9.4f}   {s_pac_t:9.4f}", flush=True)
    print(f"real                        {r_pac_f:9.4f}   {r_pac_t:9.4f}", flush=True)
    print(f"phase circular std: sim={np.sqrt(-2*np.log(np.abs(np.mean(np.exp(1j*s_ph))))):.3f}  "
          f"real={np.sqrt(-2*np.log(np.abs(np.mean(np.exp(1j*r_ph))))):.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    pbins = np.linspace(-np.pi, np.pi, 80)
    ax.hist(s_ph, bins=pbins, density=True, alpha=0.55, label='sim')
    ax.hist(r_ph, bins=pbins, density=True, alpha=0.55, label='real')
    ax.set_xlabel('phase (rad)'); ax.set_ylabel('density')
    ax.set_title('phase distribution (unflagged)'); ax.legend()
    plt.tight_layout(); plt.savefig(out / 'phase_dist.png', dpi=120); plt.close()

    # ---------- 4. MASK / RFI MORPHOLOGY ----------
    log("mask morphology...")
    s_prof = flag_band_profile(s_mask)
    r_prof = flag_band_profile(r_flags)
    print("\n=== FLAG MORPHOLOGY ===", flush=True)
    print(f"flag frac: sim={s_mask.mean():.3f}  real={r_flags.mean():.3f}", flush=True)
    # per-sample flag frac spread
    print(f"per-sample flag frac: sim {s_mask.mean(axis=(1,2)).min():.2f}-{s_mask.mean(axis=(1,2)).max():.2f}  "
          f"real {r_flags.mean(axis=(1,2)).min():.2f}-{r_flags.mean(axis=(1,2)).max():.2f}", flush=True)

    persistent = [(930, 960), (1170, 1300), (1525, 1630)]
    print("\nflag fraction inside known persistent bands:", flush=True)
    for lo, hi in persistent:
        sel = (freqs >= lo) & (freqs <= hi)
        print(f"  {lo}-{hi} MHz: sim={s_prof[sel].mean():.3f}  real={r_prof[sel].mean():.3f}", flush=True)

    # contiguous band-width distribution along freq (per time row)
    def band_widths(masks):
        w = []
        for im in masks:
            for row in im:
                d = np.diff(np.concatenate([[0], (row > 0.5).astype(int), [0]]))
                starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
                w.extend((ends - starts).tolist())
        return np.array(w) if w else np.array([0])
    sw = band_widths(s_mask[:nstruct]); rw = band_widths(r_flags[:nstruct])
    print(f"\ncontiguous freq-band width (px): sim median={np.median(sw):.0f} p90={np.percentile(sw,90):.0f}  "
          f"real median={np.median(rw):.0f} p90={np.percentile(rw,90):.0f}", flush=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(freqs, r_prof, label='real flags', lw=1.2)
    ax.plot(freqs, s_prof, label='sim mask', lw=1.2, alpha=0.8)
    for lo, hi in persistent:
        ax.axvspan(lo, hi, color='grey', alpha=0.15)
    ax.set_xlabel('freq (MHz)'); ax.set_ylabel('flag fraction')
    ax.set_title('flag fraction vs frequency (grey = known persistent bands)')
    ax.legend()
    plt.tight_layout(); plt.savefig(out / 'flag_profile.png', dpi=120); plt.close()

    # ---------- side-by-side waterfalls ----------
    log("example waterfalls...")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    vmax = np.percentile(r_clean, 99)
    for j in range(3):
        si = j * (s_amp.shape[0] // 3)
        ri = j * (r_amp.shape[0] // 3)
        sm = np.ma.array(s_amp[si], mask=~s_valid[si])
        rm = np.ma.array(r_amp[ri], mask=~r_valid[ri])
        ax[0, j].imshow(sm.T, aspect='auto', origin='lower', vmin=0, vmax=vmax,
                        cmap='plasma', extent=[0, 512, fmin, fmax])
        ax[0, j].set_title(f'sim #{si}'); ax[0, j].set_ylabel('freq MHz')
        ax[1, j].imshow(rm.T, aspect='auto', origin='lower', vmin=0, vmax=vmax,
                        cmap='plasma', extent=[0, 512, fmin, fmax])
        ax[1, j].set_title(f'real #{ri}'); ax[1, j].set_xlabel('time')
    plt.tight_layout(); plt.savefig(out / 'waterfalls.png', dpi=110); plt.close()

    log(f"done. plots in {out}/")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', nargs='+',
                    default=['/scratch3/users/jfmalan/rfi/simulated/run*/dataset.h5'])
    ap.add_argument('--real',
                    default='/scratch3/users/jfmalan/rfi/real/variants/v1_upsample512.h5')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--output', default='data_preparation/real/vis-real/sim_real_gap')
    main(ap.parse_args())
