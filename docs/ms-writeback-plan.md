# MS write-back: scope and design

Turning inpainted patches back into a Measurement Set — the inverse of extraction
(`extract_patches_sim.py` / `extract_ms.py`). This is the unbuilt piece that gates the
end goal "inpaint real measurement sets" and the image-domain evaluation.

Written 2026-06-23. Status: first implementation in `inference/`, **UNVALIDATED — must pass
the sim round-trip (below) before being run on real data.**

## The forward transform (what we must invert)

Per baseline (sim) or per (baseline, run) (real), extraction did:
1. **Channel crop** to `[freq_min, freq_max]` (900–1650 MHz) → native channels `chan_lo:chan_hi`
   (`chan_lo` stored in attrs). Channels outside the band are never modelled.
2. **Pol reduction**: `amp = |DATA|.mean(pol)`, `phase = angle(DATA.mean(pol))`,
   `flag = FLAG.any(pol)`. One amp + one phase per (time, channel); per-pol detail is gone.
3. **Divisive normalisation** (native res): `norm = wf / divisor`, `divisor` = 64-bin
   freq-smoothed bandpass. Both stored.
4. **Resize** native `(n_time_unit, n_chan)` → `512×512` (skimage `resize`, order=1, anti-alias).

Row layout (critical): the MS is assumed **time-major, fixed baseline order, no gaps**:
`row = (time_lo + t) * n_baseline + baseline_id`. `n_baseline = n_row // n_time` exactly as
extraction computed it. Sim: `time_lo=0`, full block. Real: per-run `time_lo`/`run_len` stored.

## The inverse (write-back), per unit

1. Run the model on the stored `512×512` (amp, cos φ, sin φ), conditioning hidden on the
   hole mask (`flags` real / `mask` sim), `noise_floor='auto'` to restore texture → inpainted
   `(amp_norm, cos, sin)` at 512.
2. **Inverse-resize** `amp_norm`, `dn_divisor`, `cos`, `sin`, hole-mask back to native
   `(n_time_unit, n_chan)`. Divisor round-trips near-losslessly because it is smooth (low-freq);
   the hole fill carries resize error but it is synthetic anyway.
3. **Recombine physical visibility**: `V = amp_norm·divisor · exp(i·atan2(sin,cos))`,
   shape `(n_time_unit, n_chan)`.
4. **Place** into MS rows `(time_lo+t)*n_baseline + baseline_id`, channels `chan_lo:chan_hi`,
   **only at hole pixels**, broadcast across pols. Known pixels keep the original data.
5. Write to a **new column** (`INPAINTED_DATA`, copied from `DATA`) — never overwrite `DATA`.
   Optionally clear `FLAG` at filled pixels (`--unflag`) so imagers use them.

## Decisions (v1)

| # | Decision | Rationale / refinement |
|---|----------|------------------------|
| 1 | Write **only hole pixels**; keep known data | resize round-trip isn't exact; holes are synthetic so error is irrelevant there |
| 2 | Broadcast one complex `V` to **all pols** | model is pol-averaged; the hole had no trusted per-pol info. Refinement: scale by per-pol bandpass ratio from neighbours |
| 3 | New column `INPAINTED_DATA` (non-destructive) | never clobber `DATA`/`CORRECTED_DATA`. `--out-col` overridable |
| 4 | `--unflag` optional, default off | unflagging is what makes imaging use the fill, but keep it opt-in |
| 5 | Inverse-resize stored `dn_divisor` (don't recompute) | divisor is smooth → round-trip error negligible; avoids re-reading native amp |
| 6 | Re-derive `n_baseline`, `n_time` from the MS with the **same field selection** | the row map must match extraction exactly; assert time-major ordering before any write |

## Coverage / known losses (report these, don't hide them)
- Channels outside 900–1650 MHz: untouched (never modelled).
- Baselines skipped at extraction (flag frac > `max_bl_flag_frac`): not inpainted, stay flagged.
- Per-pol structure inside holes: not recovered (pol-averaged fill).
- Resize is interpolation both ways; only hole pixels are affected.

## Safety assertions (fail before writing, not after)
- `n_row % n_time == 0` and each reshaped time-block has a single timestamp (time-major check).
- Stored `native_n_chan == chan_hi - chan_lo` and `chan_lo` matches the MS band crop.
- Target row indices in range; unit's `baseline_id` consistent with `ant1/ant2` in the MS.

## Validation: sim round-trip (REQUIRED before real)
The sim MS has ground truth (`clean`), so we can measure the write-back end-to-end:
1. Take a sim run's `sim_clean.ms` + `dataset.h5` (+ a trained sim checkpoint).
2. `inpaint_ms.py --sim` → writes `INPAINTED_DATA`.
3. Re-extract from `INPAINTED_DATA` and compare the hole pixels to `clean`:
   - complex-vis MAE in holes (should beat mean-fill / interp);
   - known pixels **unchanged** (regression check: `INPAINTED_DATA == DATA` outside holes);
   - amplitude/phase sanity vs the patch the model produced.
Only after this passes do we run on a real flagged MS.

## Files
- `inference/inpaint_ms.py` — the write-back (model inference + inverse transform + MS write).
- `inference/jobs/inpaint_ms.sh` — GPU SLURM launcher.
- Validation reuses `extract_*` to re-extract `INPAINTED_DATA` + a small compare step.

## Open questions for later
- Per-pol fill (decision 2 refinement) — only if image-domain results show pol artefacts.
- Multiple SPWs / multiple fields — current extraction assumes one SPW, single field; write-back
  inherits that assumption. Document the dataset's SPW/field layout before running.
