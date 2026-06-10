# Simulated data preparation

Two stages produce the training `dataset.h5`:

1. `extract_patches_sim.py` — reads an MS, builds a per-baseline amplitude waterfall, divisively
   normalises it, and tiles it into `(256, 256)` (time × freq) patches. Writes the clean patches,
   the phase, the normalisation divisor, and the position metadata.
2. `inject_rfi.py` — loads that file, adds RFI to the amplitude of each patch, and writes `corrupted`
   and `mask`. All other datasets and attrs are passed through unchanged.

## dataset.h5 schema
All patch datasets are `(N, 256, 256)` float32 (time × freq) unless noted; `N = n_patches`.

| Dataset | Shape | Type | Contents |
|---------|-------|------|----------|
| `clean` | `(N, 256, 256)` | f32 | divisively-normalised amplitude (unit scale, mean≈1) |
| `corrupted` | `(N, 256, 256)` | f32 | `clean` + injected RFI (amplitude only) |
| `mask` | `(N, 256, 256)` | f32 | RFI mask, 1 = corrupted pixel |
| `phase` | `(N, 256, 256)` | f32 | per-pixel phase (radians) of the patch |
| `dn_divisor` | `(N, 256, 256)` | f32 | the smoothing curve used by divisive norm |
| `freq_min_patch` | `(N,)` | f32 | low frequency edge of the patch (MHz) |
| `freq_max_patch` | `(N,)` | f32 | high frequency edge of the patch (MHz) |
| `chan_offset` | `(N,)` | i32 | first channel of the patch in the full MS (`chan_lo + o`) |
| `time_offset` | `(N,)` | i32 | first time row of the patch in the waterfall |
| `baseline_id` | `(N,)` | i32 | baseline index the patch came from |
| `ant1` | `(N,)` | i32 | first antenna of that baseline |
| `ant2` | `(N,)` | i32 | second antenna of that baseline |

Attrs: `freq_min_mhz`, `freq_max_mhz` (band edges), `n_time`, `n_freq` (patch size, 256/256),
`n_patches`, `full_n_time`, `full_n_chan` (waterfall size before tiling), `chan_lo` (first channel of
the selected band in the MS), `n_baseline`. `inject_rfi.py` also adds `seed`.

## Why the extra fields exist
The model only needs `clean`, `corrupted`, `mask`, `phase` for training. The rest support turning a
reconstructed patch back into a complex visibility and writing it into an MS:

- **`phase`** — the model reconstructs cos/sin of this alongside amplitude, so a patch becomes a complex
  number `A · e^{iφ}` rather than amplitude-only.
- **`dn_divisor`** — divisive normalisation divides each amplitude row by a smoothed bandpass estimate.
  Multiplying the (normalised) amplitude by `dn_divisor` inverts it back to physical Jy.
- **`chan_offset` / `time_offset` / `baseline_id` / `ant1` / `ant2`** — locate the patch in the original
  MS (which baseline, which time/channel block) so the inpainted result can be written to the right rows.

## Phase/amplitude reduction (known approximation)
The MS has multiple polarisations per visibility. The waterfall reduces over polarisation as:

- amplitude: `mean(|V|)` over the polarisation axis,
- phase: `angle(mean(V))` over the polarisation axis.

These are not a matched pair — the amplitude is a mean of magnitudes while the phase is the angle of a
(complex) mean — so the reconstructed `A · e^{iφ}` is an approximation of the per-polarisation
visibilities, adequate for write-back but not exact. Documented here so it is not mistaken for a bug.
