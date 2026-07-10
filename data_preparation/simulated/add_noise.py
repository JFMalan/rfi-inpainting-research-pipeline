import sys
import time
import numpy as np

pos = [a for a in sys.argv[1:] if not a.startswith('-')]
if not pos:
    raise RuntimeError("MS path not found in argv")
ms_path = pos[0]
noise_seed = int(pos[1]) if len(pos) > 1 else 0
noise_scale = float(pos[2]) if len(pos) > 2 else 1.0   # 0 = noise-free; 1 = physical MeerKAT SEFD; 2 = 2x, etc.
sefd_spec = pos[3] if len(pos) > 3 else ''             # "mhz:jy mhz:jy ..." from the telescope config
dump_t = float(pos[4]) if len(pos) > 4 else 8.0

CHUNK = 50000


def copy_column(src, dst):
    tb.open(ms_path, nomodify=False)
    if dst not in tb.colnames():
        desc = tb.getcoldesc(src)
        desc['comment'] = 'pre-noise clean signal snapshot'
        tb.addcols({dst: desc})
    nrow = tb.nrows()
    for start in range(0, nrow, CHUNK):
        n = min(CHUNK, nrow - start)
        tb.putcol(dst, tb.getcol(src, startrow=start, nrow=n), startrow=start, nrow=n)
    tb.close()
    print(f"copied {src} -> {dst} ({nrow} rows)", flush=True)


tb.open(ms_path + '/SPECTRAL_WINDOW')
cf = tb.getcol('CHAN_FREQ')            # casatools shape: (nchan, nrow)
chan_freqs = cf[:, 0] if cf.ndim == 2 else cf.ravel()   # Hz, all channels
delta_nu = float(tb.getcol('CHAN_WIDTH')[0, 0])
tb.close()

# MeerKAT L-band SEFD profile (Jy) — piecewise linear over freq (MHz)
# Values from MeerKAT array release paper (Mauch et al. 2020) and
# commissioning sensitivity measurements. Overridable via argv[4] with the
# telescope-config profile so the dump time and SEFD have one source of truth.
_SEFD_NODES_MHZ = np.array([856,  900,  950, 1000, 1100, 1280, 1400, 1450, 1550, 1600, 1650, 1712])
_SEFD_NODES_JY  = np.array([560,  510,  450,  420,  390,  390,  400,  420,  450,  470,  500,  560])
if sefd_spec:
    pairs = [p.split(':') for p in sefd_spec.split()]
    _SEFD_NODES_MHZ = np.array([float(m) for m, _ in pairs])
    _SEFD_NODES_JY = np.array([float(j) for _, j in pairs])
    print(f"SEFD profile from config: {len(pairs)} nodes  dump_t={dump_t}s", flush=True)

freq_mhz = chan_freqs / 1e6
sefd_per_chan = np.interp(freq_mhz, _SEFD_NODES_MHZ, _SEFD_NODES_JY)  # (nchan,)

delta_t = dump_t
sigma_per_chan = noise_scale * sefd_per_chan / np.sqrt(2.0 * delta_nu * delta_t)  # (nchan,)
print(f"noise_scale={noise_scale}  (0=noise-free, 1=physical MeerKAT)  "
      f"nchan={len(chan_freqs)}", flush=True)

# clean-target snapshot: DATA is still the pure crystalball signal here; sm.corrupt
# writes noise into BOTH DATA and CORRECTED_DATA, so this is the last clean moment.
copy_column('DATA', 'CLEAN_DATA')

if noise_scale <= 0:
    # extract reads CORRECTED_DATA when present, but crystalball wrote the clean signal to
    # DATA and no sm.corrupt runs here, so CORRECTED_DATA would be empty (zeros). Copy DATA
    # across so the noise-free extraction picks up the clean signal.
    print("noise_scale=0 -> noise-free; copying DATA to CORRECTED_DATA", flush=True)
    tb.open(ms_path, nomodify=False)
    has_corr = 'CORRECTED_DATA' in tb.colnames()
    tb.close()
    if has_corr:
        copy_column('DATA', 'CORRECTED_DATA')
    sys.exit(0)

sigma_mean = sigma_per_chan.mean()
sigma_min  = sigma_per_chan.min()
sigma_max  = sigma_per_chan.max()
print(f"delta_nu={delta_nu:.1f} Hz  sigma mean={sigma_mean:.4f} Jy  "
      f"min={sigma_min:.4f}  max={sigma_max:.4f}  ms={ms_path}")

# sm.setnoise only accepts a single sigma value, so we add the frequency-
# dependent noise directly via the tb tool after sm.corrupt adds flat noise.
# Strategy: use sm.corrupt for the flat baseline noise at the mean sigma,
# then add the residual per-channel variance on top.
sigma_flat    = float(sigma_mean)
sigma_residual = np.sqrt(np.maximum(sigma_per_chan**2 - sigma_flat**2, 0.0))

sm.openfromms(ms_path)
sm.setnoise(mode='simplenoise', simplenoise=f'{sigma_flat}Jy')
sm.corrupt()
sm.close()

# Add per-channel residual noise to reach the target sigma at each channel, applied
# identically to DATA and CORRECTED_DATA (sm.corrupt fills both; extraction reads
# CORRECTED_DATA, imaging uses DATA — they must stay the same visibilities).
if sigma_residual.max() > 0.001 * sigma_mean:
    tb.open(ms_path, nomodify=False)
    noisy_cols = [c for c in ('DATA', 'CORRECTED_DATA') if c in tb.colnames()]
    nrow = tb.nrows()
    rng = np.random.default_rng(seed=noise_seed)
    t0 = time.time()
    print(f"adding per-channel residual noise to {nrow} rows ({'+'.join(noisy_cols)})", flush=True)
    sig = sigma_residual[np.newaxis, :, np.newaxis]
    for start in range(0, nrow, CHUNK):
        n = min(CHUNK, nrow - start)
        noise = None
        for col in noisy_cols:
            d = tb.getcol(col, startrow=start, nrow=n)
            if noise is None:
                noise = ((rng.normal(0.0, 1.0, d.shape) + 1j * rng.normal(0.0, 1.0, d.shape))
                         * sig).astype(np.complex64)
            tb.putcol(col, d + noise, startrow=start, nrow=n)
            del d
        del noise
        rate = (start + n) / max(time.time() - t0, 1e-6)
        print(f"  rows {start + n}/{nrow}  ({rate:.0f} rows/s)", flush=True)
    tb.close()
    print(f"added per-channel residual noise  max_residual_sigma={sigma_residual.max():.4f} Jy", flush=True)
