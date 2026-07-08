import sys
import numpy as np

pos = [a for a in sys.argv[1:] if not a.startswith('-')]
if not pos:
    raise RuntimeError("MS path not found in argv")
ms_path = pos[0]
noise_seed = int(pos[1]) if len(pos) > 1 else 0
noise_scale = float(pos[2]) if len(pos) > 2 else 1.0   # 0 = noise-free; 1 = physical MeerKAT SEFD; 2 = 2x, etc.

tb.open(ms_path + '/SPECTRAL_WINDOW')
chan_freqs = tb.getcol('CHAN_FREQ')[0]   # Hz, shape (nchan,)
delta_nu   = float(tb.getcol('CHAN_WIDTH')[0, 0])
tb.close()

# MeerKAT L-band SEFD profile (Jy) — piecewise linear over freq (MHz)
# Values from MeerKAT array release paper (Mauch et al. 2020) and
# commissioning sensitivity measurements.
_SEFD_NODES_MHZ = np.array([856,  900,  950, 1000, 1100, 1280, 1400, 1450, 1550, 1600, 1650, 1712])
_SEFD_NODES_JY  = np.array([560,  510,  450,  420,  390,  390,  400,  420,  450,  470,  500,  560])

freq_mhz = chan_freqs / 1e6
sefd_per_chan = np.interp(freq_mhz, _SEFD_NODES_MHZ, _SEFD_NODES_JY)  # (nchan,)

delta_t = 8.0
sigma_per_chan = noise_scale * sefd_per_chan / np.sqrt(2.0 * delta_nu * delta_t)  # (nchan,)
print(f"noise_scale={noise_scale}  (0=noise-free, 1=physical MeerKAT)", flush=True)

if noise_scale <= 0:
    # extract reads CORRECTED_DATA when present, but crystalball wrote the clean signal to
    # DATA and no sm.corrupt runs here, so CORRECTED_DATA would be empty (zeros). Copy DATA
    # across so the noise-free extraction picks up the clean signal.
    print("noise_scale=0 -> noise-free; copying DATA to CORRECTED_DATA", flush=True)
    tb.open(ms_path, nomodify=False)
    if 'CORRECTED_DATA' in tb.colnames():
        nrow = tb.nrows()
        chunk = 50000
        for start in range(0, nrow, chunk):
            n = min(chunk, nrow - start)
            tb.putcol('CORRECTED_DATA', tb.getcol('DATA', startrow=start, nrow=n),
                      startrow=start, nrow=n)
        print(f"copied DATA -> CORRECTED_DATA ({nrow} rows)", flush=True)
    tb.close()
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

# Add per-channel residual noise to reach the target sigma at each channel,
# in row chunks so peak RAM stays bounded regardless of synthesis length / nchan.
if sigma_residual.max() > 0.001 * sigma_mean:
    import time
    tb.open(ms_path, nomodify=False)
    nrow = tb.nrows()
    rng = np.random.default_rng(seed=noise_seed)
    chunk = 50000
    t0 = time.time()
    print(f"adding per-channel residual noise to {nrow} rows in chunks", flush=True)
    for start in range(0, nrow, chunk):
        n = min(chunk, nrow - start)
        d = tb.getcol('DATA', startrow=start, nrow=n)   # (npol, nchan, n)
        sig = sigma_residual[np.newaxis, :, np.newaxis]
        noise = (rng.normal(0.0, 1.0, d.shape) + 1j * rng.normal(0.0, 1.0, d.shape))
        d += (noise * sig).astype(np.complex64)
        tb.putcol('DATA', d, startrow=start, nrow=n)
        del d, noise
        rate = (start + n) / max(time.time() - t0, 1e-6)
        print(f"  rows {start + n}/{nrow}  ({rate:.0f} rows/s)", flush=True)
    tb.close()
    print(f"added per-channel residual noise  max_residual_sigma={sigma_residual.max():.4f} Jy", flush=True)
