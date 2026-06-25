# Handover: RFI-inpainting → imaging investigation

Written 2026-06-25. For the next agent picking up the "does inpainting beat flagging for imaging"
question. Read this fully before touching the write-back or imaging code.

---

## 0. TL;DR and the central open puzzle

**Goal:** train a conditional diffusion model to inpaint RFI-flagged MeerKAT visibilities
(amplitude + phase), write the recovered complex visibilities back into a Measurement Set, and
test whether the resulting sky image is better than simply flagging the RFI.

**Where we are:** the per-patch model results are real (it recovers the smooth/coherent component
and phase). But every imaging test so far shows **inpainted worse than flagged**, and we have now
isolated *why this is suspicious*:

> **THE PUZZLE: an oracle write-back — writing the TRUE clean visibility into the holes — also
> loses to flagging** (off-source RMS: clean 1.262e-4, flagged 2.461e-4, **oracle 3.107e-4**).
> If the write-back path were lossless, the oracle image would *equal* the clean image. It doesn't.
> So a perfect fill is being corrupted by the pipeline. **Find and fix that corruption before
> trusting any imaging number.** Section 4 ranks the suspects; the phase-angle-resize one is new
> and untested.

Until the oracle reproduces the clean image (or we understand exactly why it can't), no
inpaint-vs-flag imaging verdict is trustworthy.

---

## 1. How the pipeline works (end to end)

### Simulated data (`data_preparation/simulated/`)
`jobs/simulate.sh` runs, in order:
1. `simms` → empty MeerKAT MS (`sim_clean.ms`), 8s dumps, L-band.
2. `crystalball` → predict a sky model (point sources) into `DATA`.
3. `add_noise.py` (CASA) → add **physical thermal noise** (MeerKAT SEFD profile) to `DATA`.
   **The MS `DATA` is RFI-FREE.**
4. `extract_patches_sim.py` → per-baseline waterfalls (time×freq), divisively normalised,
   resized to 512×512, written to `dataset.h5` (keys: `clean`, `phase`, `mask`?, `dn_divisor`,
   `baseline_id`, `native_n_time`, `native_n_chan`, attrs `chan_lo`, `full_n_chan`, ...).
5. `inject_rfi.py` → adds synthetic RFI into the **h5 only** (`corrupted` = clean+RFI, `mask`).
   **RFI is never written to the MS.** (This is fine — see §4, it does NOT invalidate the test.)

### Real data (`data_preparation/real/`)
- `jobs/flag_real.sh` → tricolour flags an MS.
- `extract_ms.py` (single dataset) or `extract_variants.py` (5 variants v1–v5) → per-(baseline,run)
  512×512 waterfalls → h5. **Two different schemas** — see the gotcha in §7.
- Real h5 keys: `data`, `phase`, `flags`, `dn_divisor`, `baseline_id`, `time_lo`,
  `native_n_time`, `native_n_chan` (the last two were ADDED 2026-06-25, commit d4604c5 — the
  variant extractor was missing them and crashed the write-back).

### Model (`model/`)
- Palette-style conditional U-Net (`unet.py`), DDPM (`diffusion.py`), 104.7M params, 512×512.
- **3 channels: amplitude (divisively normalised), cos(phase), sin(phase).**
- `data.py`: `PatchDataset` (sim), `RealDataset` (real, self-supervised mixed masking),
  `smooth_component()` (the low-pass "recoverable" target), `divisive_norm`, `build_cond`.
- Training: `train.py` (sim, selects best.pt by `complex_mae`), `train_real.py` (real,
  self-supervised; selects best.pt by `complex_mae` — was TRE, fixed this session).
- Sampling: DDIM `eta=0` (deterministic). `noise_floor`: `None` = smooth conditional-mean fill
  (use for imaging), `'auto'` = resampled texture (use for single-baseline waterfall viz ONLY;
  it HURTS imaging — it injects noise that doesn't average down).

### Write-back (`inference/`) — two-stage because of container split
GPU container has torch but NOT casacore; ASTRO-PY3.10 has casacore but no torch.
- `inpaint_infer.py` (GPU): run the model per unit → save preds `.npz` (cap, 3, 512, 512).
  `--oracle` skips the model and writes the TRUE clean visibility (the puzzle probe).
- `inpaint_write.py` (CPU): per unit, inverse-resize 512→native, recombine
  `V = amp·divisor·exp(i·atan2(sin,cos))`, write into a new column (`INPAINTED_DATA`) at the
  hole pixels only, broadcast to all pols. `--weight-frac F` writes the fill into
  `WEIGHT_SPECTRUM` at `F×` the real weight instead of full unit weight (down-weighting).
  `--unflag` clears FLAG at filled pixels.
- `jobs/inpaint_ms.sh`: orchestrates both stages. Env knobs: `SIM`, `SMOOTH`, `NOISE_FLOOR`
  (default `none`), `STEPS`, `WEIGHT_FRAC`, `OUTCOL`, `ORACLE`, `MAX_UNITS`, `CKPT`, `MS`, `H5`.

### Imaging evaluation (`evaluation/`)
- `image_eval.sh`: images three ways with wsclean (`oxkat-0.41.sif`): **clean** (`DATA`),
  **flagged** (`DATA`, holes flagged via `set_holes_flag.py`), **inpainted** (`INPAINTED_DATA` /
  `INPCOL`, holes unflagged). Then `compare_images.py`.
- `compare_images.py`: off-source RMS (sigma-clipped), dynamic range, and image RMSE-vs-clean.
- `set_holes_flag.py`: toggles FLAG at the h5 hole locations (`set`/`clear`).

### Cross-baseline recoverability probe (`data_preparation/real/`)
- `cross_baseline_recoverability.py` + `jobs/cross_baseline.sh`: measures whether a flagged
  visibility is predictable from uv-neighbour baselines (cross-baseline R²) vs ±1 freq channel
  (within-baseline R²) vs the per-baseline time-mean (mean-fill). Reads a real MS read-only.

---

## 2. The three trained models
1. **`phase1_all_old`** — sim, full-amplitude target. `…/runs/phase1_all_old/best.pt`. Backed up
   so retraining couldn't overwrite it. Seeds the real finetune.
2. **`v5_all512_finetune`** — real, decompose (smooth-target), init from #1.
   `…/runs/phase2_decompose_oldinit/v5_all512_finetune/best.pt`.
3. **`v5_all512_scratch`** — real, decompose, random init.
   `…/runs/phase2_decompose_oldinit/v5_all512_scratch/best.pt`.
Both real models beat interp + mean-fill on the per-patch smooth target (finetune slightly better).

---

## 3. What we learned (verified)

- **Per-patch, the model works** on the *recoverable* target: real test fake-hole MAE 0.057 <
  interp 0.080 < mean-fill 0.066 (vs the smooth component). Phase recovers. Sim prior helps
  (finetune complex_mae 0.355 < scratch 0.411). These are publishable method results.
- **An independent 13-agent audit** (2026-06-24) concluded: CONDITIONAL, not a wall, but the
  project had been pointed at the worst arena (bright-source continuum RMS) and several framing
  claims were stale/wrong. See `docs/refactor-plan.md` Progress and the audit's recorded findings.
- **The "86% irreducible noise" framing is stale and mis-basis'd.** Live code (sigma=1.0) says
  ~67% irreducible / ~33% recoverable. And it was measured *within a single baseline*; imaging
  integrates *across* baselines, where per-baseline thermal noise averages down. So the
  within-baseline noise floor does NOT bound the imaging benefit.
- **Cross-baseline probe ran** (2026-06-24, on the real MS, raw `DATA` and `CORRECTED_DATA` —
  which were near-identical, i.e. effectively uncalibrated): complex-V cross-baseline R² = −0.09,
  amplitude −0.39, within-baseline −0.02/−0.16, uv-coherence ~0.1. **Negative R² = a uv-neighbour
  predictor is worse than the per-baseline time-mean.** CAVEATS (important, see last session):
  this is a bright **calibrator** field (mean-fill near-optimal by construction), **uncalibrated**
  (`CORRECTED_DATA` ≈ `DATA`, identical noise proxy), and it tests predictors the model doesn't use.
  So it corroborates "raw residual is noise" but is NOT a clean ceiling verdict.
- **The sim imaging test is NOT invalid** (correcting an earlier audit claim). The three images
  are invariant to whether RFI is in the MS: clean grids all clean DATA, flagged drops holes,
  inpainted overwrites holes with the model fill (which comes from the h5, not the MS). So
  injecting RFI into the MS would give the identical verdict — don't build that. The test is valid;
  the result reflects fill quality + weighting + (we now suspect) a path bug.

---

## 4. THE CENTRAL PUZZLE: why does the oracle (true fill) lose to flagging?

`inpaint_infer.py --oracle` writes the TRUE clean amplitude (h5 `clean`) + phase back into the
holes. Expectation: oracle inpainted image ≈ clean image (it puts the true values back). Result:

```
clean    off-src RMS 1.262e-4   DR 13648
flagged              2.461e-4   DR 6948
oracle               3.107e-4   DR 5435   ← should be ≈ clean, but it's WORSE than flagged
```

A perfect fill is being corrupted by the write-back path. **This is the bug to find.** Candidate
causes, ranked by my current suspicion, each with a concrete diagnostic:

### Suspect #1 (NEW, untested): phase is resized as a raw ANGLE → 2π-wrap corruption
`extract_patches_sim.py` / `extract_ms.py` / `extract_variants.py` all store
`phase = resize(np.angle(DATA.mean(pol)))` — they **resize the wrapped phase angle** (native→512)
with linear interpolation. At a wrap (e.g. +3.1 → −3.1 rad), linear interp gives ~0, which is
totally wrong. Off-phase-centre sources produce fringes → many wraps across the band → corrupted
stored phase. The oracle then reconstructs `V = amp·exp(i·phase)` from this corrupted phase →
visibilities point the wrong way → coherent image error. **The model is also trained on this
corrupted phase.** 
- **Fix:** resize `cos(phase)` and `sin(phase)` (smooth, no wrap), or resize the complex
  visibility, NOT the raw angle. This is an extraction change → re-extract + retrain.
- **Diagnostic (cheap):** see the layered oracle below — if the native-true oracle (Level 0)
  reproduces clean but the h5-phase oracle (Level 1) doesn't, phase-angle-resize is implicated.

### Suspect #2: polarisation collapse
Extraction reduces pols inconsistently (`amp = mean(|XX|,|YY|)`, `phase = angle(mean(XX,YY))`),
and the write-back writes ONE V to ALL pols (`XX=YY`). For an unpolarised field this is small, but
it is non-invertible and injects per-pol error. (`inpaint_write.py` `band[:,:,p] = where(hole,V,band)`.)

### Suspect #3: divisor double-resize
The divisive-norm divisor is resized native→512 (storage) then 512→native (write-back). It's smooth
so this should be near-lossless, but verify (it directly scales reconstructed amplitude).

### Suspect #4: full-weight hard substitution
The oracle ran at full unit weight (no `--weight-frac`). A fill at full weight injects its error at
the same imaging weight as a real measurement. Even a small per-pixel error, at full weight over
~180k pixels/baseline, can exceed flagging's coverage penalty. **The weight sweep (below) tests this.**

### Suspect #5: 512-resize round-trip
897 freq channels → 512 → 897 blurs the hole content. Smooth fills survive; the true noisy
amplitude does not. Probably second-order vs #1/#4.

### The decisive diagnostic: a LAYERED oracle
Run these in order; the first one that reproduces the clean image localises the loss:
- **Level 0 — native passthrough:** read the EXACT MS `DATA` at the hole rows/channels and write it
  straight back to `INPAINTED_DATA` (no h5, no resize, no pol-collapse, no divisor). Image it.
  - If Level 0 ≠ clean → the bug is in the **row/channel/pol mechanics** of the write-back
    (`sr = time_lo*n_baseline+bl`, `rowincr`, `chan_lo:chan_hi`, pol loop). Fix that first.
  - If Level 0 == clean → mechanics are fine; the loss is in the **h5 representation**.
- **Level 1 — h5 representation oracle (current `--oracle`):** if this loses but Level 0 wins, the
  loss is resize + pol-collapse + phase-angle. Then bisect: try a corrected phase (resize cos/sin),
  then per-pol, then native-res, to see which recovers the clean image.

Level 0 is the single most informative next experiment and is cheap to build (a ~30-line script).

### The weight sweep (tests Suspect #4, also cheap, reuses saved oracle preds)
Re-run `inpaint_write.py` with the saved oracle preds and `--weight-frac ∈ {0.05, 0.2, 0.5}`, image
each. If some weight makes the oracle beat flagged, full-weight substitution was a major factor and
weighting is part of the fix. If even the best weight only ties flagged (as W→0), the fill adds no
imaging value in this representation.

---

## 5. What to do next (priority order)
1. **Build the Level-0 native-passthrough oracle and image it.** This splits "write-back mechanics
   bug" from "representation loss" — the fastest way to crack the puzzle. (No model, no GPU.)
2. If mechanics are fine → **test the phase-angle-resize fix** (Suspect #1): a corrected oracle that
   resizes cos/sin. If that recovers the clean image, the phase pipeline is the bug and needs an
   extraction + retrain fix.
3. **Weight sweep** on the oracle (Suspect #4) — cheap, reuses saved preds.
4. Only once the oracle reproduces clean (or we understand the floor): re-test the actual model fill,
   then decide if continuum imaging is winnable or pivot to the power-spectrum/spectral domain
   (where the literature says RFI-gap inpainting actually helps).

Do NOT report any inpaint-vs-flag imaging conclusion until the oracle puzzle is resolved.

---

## 6. Key numbers / state
- Sim imaging (run1, full-amp model, none-fill): inpainted RMS 3.751e-4; auto-fill 4.235e-4;
  **oracle 3.107e-4**; flagged 2.461e-4; clean 1.262e-4.
- Real write-back: stage-1 inference done (preds saved at
  `/scratch3/users/$USER/rfi/inpaint_preds_71796.npz`, 2936 units); stage 2 re-runnable.
  `v5_all512.h5` re-extracted with native keys, 2936 samples (matches → preds align).
- Real MS: `/idia/projects/astro-cirg/data_for_rfi/1570802018_sdp_l0-J2018_5539-corr.ms`
  (raw `sdp_l0`, 30 cols, 2226011 rows, 1301 timestamps, 1711 baselines, two good time-runs
  35–332 and 410–781). Copy to scratch before any write-back (it's shared/read-only project data).

---

## 7. Gotchas (will bite you)
- **Singularity is blocked on the login node** (user-namespace error). Run via `srun`/`sbatch` only.
- **Push to `main` is blocked by the safety classifier** for the agent — commit locally; the USER
  pushes. Don't try to work around it.
- **`/scratch3` auto-deletes** files unused for 90 days (4th Tuesday monthly).
- **Container split:** torch only in `ASTRO-GPU-PyTorch-2026-01-28.sif`; casacore+skimage only in
  `ASTRO-PY3.10.sif`; wsclean in `oxkat-0.41.sif`; CASA for `add_noise.py`. Hence the two-stage
  write-back.
- **casacore column-add** needs `maketabdesc` (NOT `maketabledesc`) + a `dminfo` dict with a unique
  `NAME`, else it clashes with the source column's data manager. See `inpaint_write.py:ensure_column`.
- **Two real-data extraction schemas:** `extract_ms.py` and `extract_variants.py` differ; variants
  store `time_lo`/`freq_lo` and (now) `native_n_time`/`native_n_chan`. Don't assume keys.
- **`noise_floor='auto'` is for waterfall viz only**, never for imaging (injects non-averaging noise).
- **`CORRECTED_DATA` on the real MS ≈ `DATA`** (uncalibrated l0); a "calibrated" probe needs an
  actually-calibrated MS (1GC first), which we don't currently have.
- **DDIM steps:** 200 is slow (~0.05 units/s at 512²). 50 steps is fine for imaging fills (~0.2/s).
- **GPU note:** V100 (gpu-005) is ~1.5× slower than A40/A100 for this; use `--constraint=A100|A40`
  to force the faster cards (may queue).

---

## 8. The honest scientific position
The method demonstrably recovers the smooth/coherent component and phase (per-patch). Whether that
translates to a better *image* is unresolved and currently blocked by the oracle puzzle (§4). The
literature precedent for RFI-gap inpainting helping downstream science is the **power spectrum /
delay space** (HERA/EoR), NOT continuum imaging — so even after the path bug is fixed, the
continuum-RMS metric may be the wrong arena. Keep both outcomes in view: a fixed path that lets
inpainting beat flagging (great), or a clean ground-truth-backed negative for continuum imaging
plus a pivot to the spectral domain (also a legitimate, publishable result).
