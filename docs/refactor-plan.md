# Codebase refactor plan

Written 2026-06-23 following the stochastic-inpainting and thermal-noise research session.
Describes what needs to change across the codebase and why, in priority order.

## Why a refactor

The current evaluation and sampling defaults are built around amplitude MAE as the primary
metric. The literature and our own measurements show this is wrong for noise-dominated data
(Blau & Michaeli theorem; Pagano et al. 2023). The refactor realigns the metric stack and
sampling defaults with what the science actually requires.

---

## Progress (2026-06-23)

- **Repo cleanup — DONE** (commit `433f966`). 9 superseded scripts moved to `archive/`
  (bias_diag, infer_compare, gen_sweep, sampler_sweep, viz_eta, extract_windows,
  merge_windows + their launchers); `archive/README.md` maps each to its replacement.
- **§1 metric stack — DONE** (commit `fddcfd3`). `metrics.noise_floor_ratio()` added;
  wired into `train.py` and `train_real.py` `val_eval()`; real training now selects
  `best.pt` by `complex_mae` instead of TRE (TRE penalised correct noise texture +
  scored against contaminated obs).
- **§2 sampling defaults — DONE** (commit `ad1c853`). `eval_real.py` `--noise-floor`
  flag (default none = metric path) + noise_floor_ratio in RESULTLINE; `inpaint_viz.py`
  `--noise-floor` flag; `compare_inpaint.py` `--resample-speckle` now drives
  `noise_floor='auto'` in the sampler instead of the manual post-hoc add.
- **Still open:** §1 FWD metric; §3 L2-loss experiment (needs retrain); §4 mean-fill
  tightening; §5 speckle_probe noise-floor condition; and the MS write-back path
  (unbuilt — gates the "inpaint real measurement sets" end goal).

---

## 1. Metrics (highest priority)

### What to add
**Noise floor ratio** (texture ratio): `σ_hp(inpainted) / σ_hp(known)` where σ_hp is the
std of the high-pass (5×5 uniform filter) residual. Already implemented in
`model/diagnostics/stochastic_inpaint.py`. Needs to be added to:
- `model/train.py` `val_eval()` — add alongside `complex_mae` so training curves show it
- `model/real/eval_real.py` — add to the RESULTLINE output (already has `std_ratios`)
- `model/metrics.py` — add a `noise_floor_ratio(pred, target, mask)` function

**Fréchet Wavelet Distance (FWD)** (arXiv:2312.15289, ICLR 2025): distributional fidelity,
no pretrained network, valid at 64px. For the writeup. Not needed in training loops, but
add to the final eval script.

### What to demote
**Amplitude MAE** on long-baseline/real data should be reported as a secondary/context
metric, not the headline. It is provably won by mean-fill when the hole is noise-dominated.
The headline metrics should be:
1. Complex-visibility MAE (already reported; this is the right per-pixel metric)
2. Phase angular error (already reported; the genuine recoverable contribution)
3. Noise floor ratio (new; statistical consistency)
4. Smooth-target MAE on simulated data (fair comparison: measures signal recovery)

### Files to change
- `model/metrics.py` — add `noise_floor_ratio()`
- `model/train.py` — add noise_floor_ratio to `val_eval()` output dict + JSON log line
- `model/real/eval_real.py` — add FWD + noise_floor_ratio to RESULTLINE
- `model/real/beat_meanfill.py` — add noise_floor_ratio to output

---

## 2. Sampling defaults in eval scripts (medium priority)

All eval/viz scripts currently call `diff.sample(..., eta=0.0)` and discard the
`noise_floor` parameter (which didn't exist until 2026-06-23). For scientific output, the
correct mode is `eta=0.0, noise_floor='auto'` (DDIM for signal fidelity + post-hoc noise
for statistical consistency). For diagnostic metrics (MAE vs smooth), `noise_floor=None`
is still correct (don't penalise for added noise).

**Files to update:**
- `model/real/eval_real.py` — add `noise_floor='auto'` to the sample call used for the
  visualization path (keep `noise_floor=None` for the metric path)
- `model/diagnostics/inpaint_viz.py` — add `--noise-floor auto` flag
- `model/diagnostics/compare_inpaint.py` — already has `--resample-speckle`; wire to
  `noise_floor='auto'` in the sample call instead of the manual post-process
- `model/diagnostics/visualise_samples.py` — add `noise_floor` arg

The `noise_floor='auto'` parameter is live in `model/diffusion.py` as of 2026-06-23.

---

## 3. Loss function experiment (lower priority, requires retraining)

Palette (Saharia et al. 2022) found L2 on ε gives more diverse samples than L1. Current
config uses L1 (`model/diffusion.py` loss uses `.abs()`). Switching to L2 would give:

```python
# in Diffusion.loss() and loss_phase2():
# current:
err = (pred - target).abs()
# proposed experiment:
err = (pred - target).pow(2)
```

This is a retraining experiment. Do NOT change as part of the main refactor without
running the speckle probe first to confirm it helps texture ratio without hurting smooth MAE.
Track as: `phase1_all_decompose_l2`.

---

## 4. train.py val_eval — smooth-target alignment (medium priority)

`val_eval()` currently evaluates `complex_mae` against whatever `x0` the dataset returns
(the smooth target when `smooth_target=True`). This is correct. But it does not yet log
the noise floor ratio, so training curves give no visibility into whether the model's
outputs have the right noise texture. Add `noise_floor_ratio` to the per-epoch log line
so you can see it improve (or not) during training.

Also: the mean-fill baseline in `val_eval()` computes mean over all channels/pixels but
mean-fill for amplitude should use the per-patch known-pixel mean, not the global mean.
The current implementation is approximately right but could be tightened.

---

## 5. Diagnostic cleanup (low priority)

- `model/diagnostics/speckle_probe.py` — add `noise_floor='auto'` condition alongside the
  existing `train_target=smooth` conditions. This is the probe that originally declared
  "model can't beat mean-fill" — re-running with the noise-floor resampling shows what
  the correct metric (texture ratio) says.
- Remove or archive: `model/diagnostics/gen_sweep.py`, `model/diagnostics/sampler_sweep.py`
  — these swept the old noisy-target metrics and are now superseded by `stochastic_inpaint.py`.

---

## What is NOT part of this refactor

- **Architecture changes** (multi-baseline conditioning, larger model) — separate project
- **Phase 2 real-data training** — already built; needs the metric fixes above first
- **Image-domain evaluation** (flag → inpaint → grid → image vs flag → image) — separate
  experiment, needs MS write-back pipeline. This is the most scientifically meaningful test
  and has never been run end-to-end. Flag as the outstanding experiment for the thesis.
- **RePaint U>1 resampling** — remains available via `repaint_u` arg; not a priority until
  there is evidence boundary artefacts are limiting quality
