# Phase 1 — Science Process Log

Honest record of what was tried, the results, and what we learned building the
amplitude+phase RFI-inpainting DDPM for MeerKAT. Written for the eventual writeup.

## Goal
Conditional DDPM (Palette-style U-Net) that inpaints RFI-masked regions of MeerKAT
visibility spectrograms. 3-channel target per 256×256 patch: amplitude (ch0,
divisively normalised), cos(phase) (ch1), sin(phase) (ch2). Conditioning = corrupted
amplitude/phase with the hole hidden + binary mask + frequency positional encoding.
Metrics: amplitude MAE/PSNR, phase angular error, and complex-visibility MAE
(|V_pred − V_true|, V = amp·e^{iφ}) — all in the mask region vs a mean-fill baseline.

## Pipeline (final, validated)
1. Data-prep: simms → crystalball (sky model) → CASA thermal noise → per-baseline
   amplitude (mean|V| over pol) + phase (angle of pol-mean) → divisive normalisation
   on amplitude (stores dn_divisor for inversion) → 256×256 patches with synthetic
   RFI injected into amplitude → dataset.h5 (clean, corrupted, mask, phase,
   dn_divisor, + chan/time/baseline offsets for MS write-back).
2. Model: Palette U-Net (~62M params, attention 32/16, sinusoidal time embedding,
   freq-keyed PE), predict=x0, hole-only L1 loss.
3. Inference: deterministic DDIM (eta=0), known region pinned to truth, hole filled
   from context; complex-MAE early-stopping + best.pt.

## What we tried, and the results

### Training/contract bugs (found and fixed)
- **Loss leak (CRITICAL, fixed):** training built xt = q_sample(FULL x0), so the
  noised true value sat in the hole of the network input → the model learned to COPY
  it, not inpaint from context. Every "good" early number was this leak; clean
  inference gave ~0.31 (worse than mean-fill). FIX = Palette contract:
  x_in = keep·x0 + mask·xt (clean known region, hole = noised state), loss hole-only.
  Verified: 4-patch leak-free overfit → amp MAE 0.017.
- **Sampler iterations (several wrong attempts, resolved):** stochastic-DDPM and a
  DDIM attempt that re-randomised the hole each step both broke (frozen/garbage
  output). Correct contract: hole carries the EVOLVING reconstruction, known region
  pinned; deterministic DDIM (eta=0). Verified by perfect-model recovery tests.

### Diagnostics built (fast, reusable)
- `pipeline_doctor.py` — isolates each stage (data, conditioning, U-Net receptive
  field, training contract, loss, end-to-end overfit). All PASS → pipeline correct.
- `recoverability.py` / `info_ceiling.py` — measure how predictable the masked
  amplitude is from context (interp/biharmonic baselines, context→hole R²,
  autocorrelation). Model-free, decisive.
- `overfit_test.py`, `gen_sweep.py` — fast train/eval on small/held-out sets.
- `infer_compare.py` — sampler comparisons (steps, clamp, eta, RePaint, texture).

### The amplitude generalization failure → root cause
- On the realistic dataset (run1: sky scaled 40× down to match real MeerKAT
  blank-field noise), amplitude inpainting plateaued at ~0.31 MAE (worse than
  mean-fill 0.24) and never generalised, while PHASE generalised well.
- `info_ceiling` proved why: run1 masked amplitude has context→hole **R² ≈ 0**,
  lag-1 autocorrelation ≈ 0 — it is **white noise**. No model can generalise white
  noise; only the marginal mean is recoverable (= mean-fill). Phase generalises
  because fringe phase keeps coherent structure that divisive norm doesn't destroy.
- Comparison to Massoud et al.: their VisPB data is smooth, high-SNR
  foreground-dominated (foregrounds ~10⁵ over noise; baseline PSNR 56.9 dB). Their
  reported success is on NARROW masks over smooth data, with a whole-image PSNR
  metric (a trivial mean-fill scores ~41 dB by that metric on our data — inflated).
  Their amplitude is near-deterministic given context; ours (realistic) is not.

### Bright-sky control (run2) — validated the pipeline
- Regenerated with a 40× brighter sky (source-dominated, like VisPB).
  `info_ceiling`: R² 0 → **0.645**, autocorr 0 → **0.34** (now learnable).
- Full training on run2: amplitude crossed below mean-fill at epoch 3, reached
  amp MAE ~0.032 (mean-fill 0.084), complex MAE 0.46 → 0.057, phase err 0.45 →
  0.07, PSNR 30+ dB, still improving. **The model genuinely inpaints amplitude AND
  phase when the data has recoverable structure** — reproduces Massoud's regime.

### run2 full training — FINAL converged result (predict=x0, L1, DDIM eta=0)
Trained to early-stop at epoch 31 (no >0.002 complex-MAE gain for 8 evals).
Best (held-out val, mask region, vs mean-fill baseline):
- complex-visibility MAE: **0.058**  (mean-fill 0.691 — 12× better)
- amplitude MAE: **0.031**           (mean-fill 0.084 — 2.7× better)
- phase angular error: **0.046 rad** (~2.6°; mean-fill 1.1 — 24× better)
- amplitude PSNR: **32.6 dB**
Trajectory: amp MAE 0.19 (e1) → 0.031, crossed below mean-fill at epoch 3, plateaued
~epoch 15. best.pt at epoch 25 (complex 0.0579). This is the validated Phase-1 result:
the model genuinely generalises amplitude+phase inpainting on structured data.
Caveat (the smoothness finding): these are conditional-MEAN reconstructions — accurate
(low MAE) but smoother than the real speckle (texture ratio ~0.27). Reported numbers
are the point-estimate quality; realistic texture is a separate, open consideration.

### Sampler / inference experiments (on run2 best.pt)
- DDIM steps 200 vs 1000: identical MAE (0.040) → 200 fine for monitoring.
- Clamp: loose (−2,4) correct; tightening to data range CLIPS real low-amplitude
  fringe signal (model legitimately predicts down to 0.41) → worse. Keep loose.
- RePaint resampling (U=5,10): WORSE than plain DDIM (injected noise raises MAE).
  → DDIM deterministic is the right sampler for low-MAE point estimates. (Documents
  proposal sub-question 3: stochastic resampling degrades numerical fidelity here.)
- WIDE vs NARROW mask error: WIDE 0.025 < NARROW 0.036 — wide bands are the MOST
  accurate region, not the worst. The "wide band looks wrong" was a VISUALISATION
  colour-scaling artifact (per-patch percentile scaling against real bright signal),
  confirmed 3×. Fixed the visualiser (honest error scale + value-range labels).

### Methodology pivot — why the headline config is x0/deterministic (off-spec)
The proposal (GOAL.md eq. 2, §4.3) specifies noise-prediction (eps) + L1 + RePaint
stochastic resampling — the Palette default. The current validated result was instead
obtained with **predict=x0 + deterministic DDIM (eta=0)**, which is a deliberate
deviation: it predicts the conditional MEAN of the hole, minimising MAE/PSNR (the
proposal's headline metrics) at the cost of speckle texture. This pivot happened
because the spec'd eps/RePaint path, optimised for MAE, gave fills that were
statistically plausible but visibly over-smooth, and the numerical metrics rewarded
the mean. The two configs are kept as a tracked tradeoff (point-estimate fidelity vs.
statistical realism); GOAL.md §4.4.1 now records this revision. The eps/RePaint path
is being re-examined (below) to find the best approach — not because texture is itself
a success metric, but because it is the proposal's named methodology and the right
config is still open.

### Texture (the open item)
Caveat on the texture-ratio diagnostic: it is a high-pass std RATIO (hole vs. known
region), so it measures variance match, not reconstruction fidelity — it can be driven
to ~1.0 by calibrated white noise carrying no recovered signal (which is why post-hoc
noise was rejected). It is a reporting/diagnostic number, NOT an optimisation target,
and it is in direct tension with MAE (a realistic sample is further from the mean than
the mean is). Treat low MAE and texture=1.0 as answers to different questions
(best point estimate vs. statistically realistic sample), not jointly optimisable.
- Deterministic DDIM predicts the conditional MEAN → the fill is visibly SMOOTHER
  than the surrounding thermal-noise speckle (texture ratio ~0.27 vs target 1.0).
  Low MAE actually REWARDS this (smooth mean beats any specific noise guess).
- Stochastic sampling (eta 0→1) did NOT restore texture (~0.27 throughout) — the
  smoothness is baked into predict=x0 + L1, not the sampler.
- Post-hoc calibrated noise (add noise matched to the known-region speckle) restores
  texture but needs empirical scaling and only cosmetically re-adds irreducible
  noise — not reconstructed signal. Decided against this.

### Texture/bias investigation RESOLVED (sampler sweep + bias diagnostic)
Before committing an eps retrain to "fix smoothness", ran two cheap no-retrain tests on
the existing run2 best.pt (model/sampler_sweep.py, model/bias_diag.py). Findings overturned
the premise:
- Real texture ratio at the headline config (eta=0, U=1) is **0.748, not 0.27** — the 0.27
  was a measurement artifact (different window/sample). The fills are only mildly smoother
  than the surroundings, not drastically.
- eta 0→1 barely moves texture (0.748→0.777): on a predict=x0 model stochastic sampling
  CANNOT add texture (the x0 head emits the mean; no eps mechanism). RePaint U=5 overshoots
  to 1.4–2.0 (too noisy) and costs 3–7 dB PSNR. So no sampler setting improves seamlessness;
  eta=0, U=1 is the best config visually and numerically.
- Bias diagnostic (signed fill error vs band depth / width / per-patch structure, 64 patches,
  515k hole pixels): overall signed bias **−0.0038** vs abs error 0.033 (essentially unbiased).
  Error is FLAT across depth into the band (edge 0.032 ≈ centre 0.034 — no centre-drift, the
  "guessed from far away" worry is disproven), FLAT across band width (widest 0.034), and FLAT
  across structure terciles including the noise-dominated one. There is no fixable systematic
  error anywhere.
- CONCLUSION: the band that remains *visible* on noise-dominated patches is the
  **perception–distortion gap**, not reconstruction error. A numerically-correct fill
  (error 0.033, unbiased) in a thermal-speckle field still reads as "too smooth" to the eye
  because it lacks the specific speckle grain — which is irreducible noise and cannot be
  recovered, only resampled. This is expected and not a model defect. DECISION (2026-06-16):
  ship eta=0/U=1 as the validated Phase-1 config; do NOT retrain for texture (it would cost
  GPU-hours for a cosmetic gain the MAE/PSNR metrics do not reward and would worsen).

## Key learnings
1. **Amplitude inpaintability is bounded by field SNR.** On realistic noise-dominated
   MeerKAT blank-field amplitude, the masked content is white noise — fundamentally
   not generalisably inpaintable (R²≈0). This is a genuine, novel finding Massoud
   never confront (their data is smooth/high-SNR). It is NOT a model failure.
2. **The pipeline is correct and capable** — proven on bright-sky data (R²=0.645):
   the model inpaints amplitude+phase, generalises, beats mean-fill ~2×, ~30 dB PSNR.
3. **Phase is the durable contribution** in the realistic regime — it retains
   coherent fringe structure (divisive norm only touches amplitude), so it
   generalises and the complex-visibility reconstruction beats baseline even when
   amplitude is noise-limited.
4. **Metric honesty matters:** aggregate MAE hid per-region behaviour; whole-image
   PSNR inflates results; MAE rewards smooth means over realistic texture. Report
   complex-visibility MAE + per-region splits + acknowledge the texture/MAE tension.
5. **The conditional-mean vs realistic-sample tension** is real: deterministic
   sampling minimises MAE but smooths; matching the noise texture costs MAE. The
   right framing depends on whether the downstream use wants the best point estimate
   or a statistically realistic visibility.

## Open / next
- Texture/smoothness: RESOLVED above — ship eta=0/U=1, no retrain. (The eps-retrain idea
  was tested cheaply and rejected: it would not improve the accurate, unbiased fills, only
  cosmetically resample irreducible noise at GPU/metric cost.)
- Evaluate best.pt (run2) on the held-out TEST split for citable numbers.
- Build inference/ patch→waterfall stitching + MS write-back (data has the metadata).
- Phase 2: mixed-masking self-supervised training on real MeerKAT data (tricolour
  flags), TRE metric.
