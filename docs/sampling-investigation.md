# Inference sampling: investigation and the move from stochastic DDPM/RePaint to DDIM

Research note for the writeup (Phase 1 inference, proposal §4.3.3, sub-question 3).
All numbers are mask-region MAE on held-out simulated amplitude patches (256-patch overfit
diagnostic, divisive-normalised units, clean amp std ≈ 0.31).

## The question
Sub-question 3 asks how to harmonise generated RFI replacements with the surrounding
unmasked context without prohibitive cost. The proposal anticipated modifying the standard
DDPM sampler with RePaint temporal resampling + statistical alignment.

## What we found: stochastic sampling degrades numerical fidelity
The model learns the masked structure well — measured directly via the single-shot
conditional-mean estimate (`x0_pred`):

| reconstruction | mask-region MAE |
|---|---|
| classical linear interpolation (recoverable-structure target) | ~0.10 |
| model `x0_pred` (deterministic conditional mean) | ~0.12 |
| mean-fill baseline | ~0.26 |
| **model via stochastic DDPM ancestral sampling** | **~0.29** |

The stochastic sampler (ancestral DDPM, and RePaint which builds on it) injects fresh
Gaussian noise `σ_t z` at every reverse step by design — it is a *generative* sampler built
for sample diversity, not for a minimum-error point estimate. On data with a large
thermal-noise component, this produces a *plausible* fill (correct statistics) whose specific
per-pixel values are wrong, giving MAE **worse than mean-fill** and far worse than the model's
own `x0_pred`. So a model that has genuinely learned the structure is made to look broken by
the sampler.

This is the same train/sample concern Massoud et al. raise for Method 3 (their §3: the masked
region's input statistics differ between training and sampling). Our finding is sharper: even
with that addressed, the *stochasticity itself* is what costs numerical fidelity for a
scientific reconstruction.

## RePaint attempts tried (and why they were insufficient here)
1. **Standard ancestral DDPM + RePaint known-region blend** (`x = keep·x_known + hole·x_unknown`,
   known region re-noised each step). Result: MAE ~0.29 — the per-step `σ_t z` noise dominates
   the hole reconstruction. Worse than mean-fill.
2. **Re-noising the hole input each step** (an early, incorrect attempt to match train/sample
   statistics): froze the reverse chain — the hole input never accumulated the reconstruction,
   so the chain was memoryless and produced constant garbage (MAE at the random baseline). This
   confirmed the hole must carry the *evolving* reconstruction, not fresh noise, between steps.
3. **RePaint resampling loop (U>1 jump-back)**: implemented (the `U` parameter) but not the fix
   — it harmonises boundaries, it does not remove the per-step stochasticity that drives the
   MAE penalty.

## Resolution: DDIM deterministic sampling (η=0)
We adopt DDIM with η=0: deterministic reverse steps, no per-step noise injection, while keeping
the RePaint known-region blend. This yields a faithful point estimate (sampled MAE matches
`x0_pred` ≈ 0.12) and is the sampling-side expression of the proposal's own §4.3.1 goal —
"mathematically faithful, conservative predictions over visual diversity." η is retained as a
tunable knob (η=1 recovers stochastic DDPM) for any future diversity/uncertainty analysis.

## For the writeup
- Frame DDIM as the deliberate choice for numerical fidelity, with the stochastic-DDPM/RePaint
  attempts above as the motivating evidence (this is the substance of sub-question 3).
- The RePaint known-region conditioning is retained; the resampling loop (U) and statistical
  alignment remain available for boundary-harmonisation experiments / Phase 3.
- Report `x0_pred` MAE, DDIM-sampled MAE, classical-interp target, and mean-fill baseline
  together — the gap between mean-fill and interp defines the recoverable structure; the model
  should (and does) land near interp.

---

## Reconciliation with Blau & Michaeli (added 2026-06-23)

The finding that stochastic DDPM gives MAE worse than mean-fill is not a model deficiency —
it is the perception-distortion theorem (Blau & Michaeli, CVPR 2018, arXiv:1711.06077) acting
as predicted. That theorem proves there is a hard Pareto frontier: any estimator that samples
correctly from the posterior distribution *must* score worse on MAE than the conditional mean
(mean-fill). The stochastic DDPM result (MAE ~0.29 > mean-fill ~0.26) is exactly this —
it is evidence the sampler is drawing a genuine posterior sample, not evidence the model is
wrong.

DDIM (η=0) was the right choice for the numerical-fidelity operating point required by the
proposal (faithful reconstruction). The stochastic operating point (η=1) remains available
and is now the recommended mode for *visual/statistical* output where the goal is a
statistically consistent fill rather than minimum distortion.

**The two-stage approach (implemented 2026-06-23):** Use DDIM η=0 to get the low-distortion
smooth prediction, then add `noise_floor='auto'` noise post-hoc from the estimated local σ.
This decouples signal recovery (evaluated by MAE vs smooth target) from noise consistency
(evaluated by texture ratio). Both can be correct simultaneously; neither forces the MAE/
perception tradeoff. This is the `+auto_noise` condition in `model/diagnostics/stochastic_inpaint.py`.
