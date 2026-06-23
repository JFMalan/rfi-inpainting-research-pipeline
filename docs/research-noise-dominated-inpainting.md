# How to usefully inpaint noise-dominated RFI-masked MeerKAT amplitude

Deep-research synthesis (2026-06-19). Verified against ~25 primary sources with 3-vote
adversarial checking. This consolidates the literature answer to: *given that the masked
amplitude is largely irreducible thermal/speckle noise (R2~0, lag-1 autocorr ~0.03), what
do we actually do?*

## Bottom line

Three findings are settled by the literature and by our own measurements, and they point
the same way:

1. **You cannot inpaint white noise — this is information-theoretic, not a model defect.**
   Under pixel-wise-independent noise, masked-pixel prediction *is* denoising, and the
   network can only recover the underlying clean signal, never the noise realisation
   (View-Blind-spot-as-Inpainting, arXiv:2109.04970). The HERA collaboration proved the
   same thing for interferometric visibility inpainting directly: inpainting reliably
   reconstructs the smooth/coherent component but provably cannot reconstruct thermal-noise
   modes, and the largest errors concentrate in the noise-dominated modes (Pagano et al.
   2023, MNRAS 520.5552, arXiv:2210.14927). Our speckle sweep reproduced exactly this.

2. **MAE is the wrong success criterion when the target is noise.** The perception-distortion
   tradeoff (Blau & Michaeli, CVPR 2018, arXiv:1711.06077) proves the MAE/MSE-minimising
   estimator is the conditional mean, which is necessarily the blurriest, least-realistic
   fill — *for any distortion measure*, so swapping MAE→MSE→L1 cannot escape it. "Can't beat
   mean-fill on MAE" is therefore the expected outcome, not a failure: mean-fill IS the
   distortion-optimal answer in a noise hole. Only changing the *objective* (to a
   distributional / perceptual / downstream one) changes the operating point.

3. **The field has already moved the target off per-pixel amplitude.** Every in-domain
   radio precedent evaluates in a downstream domain where noise averages down, or by
   statistical consistency — not per-pixel error.

**Recommendation: a combination of (b) and (c), not (a).** Stop optimising/evaluating
per-pixel amplitude MAE (option a is a dead end — it's provably won by mean-fill). Instead:
reformulate as **decompose-then-inpaint** (option c) — inpaint the recoverable smooth
component, resample the noise residual from its statistics — and **pivot the scientific
framing** (option b) to the complex visibility and the image/power-spectrum domain where
thermal noise integrates down. Keep the phase result front-and-centre; it is the genuine
contribution and the literature explains why it works.

---

## Ranked strategies

### 1. Decompose: inpaint the smooth/coherent component, resample the noise residual
**Strongest, most directly supported.** Split each waterfall into a smooth component
(recoverable, our std ~0.098) + a noise/texture residual (our ~86% of variance), inpaint
only the former, and *synthesise/resample* the residual from its statistics rather than
trying to predict it pixel-exactly.

Evidence:
- **DG3PD** (Thai & Gottschlich, Royal Society Open Science 2018): formal cartoon+texture+
  noise decomposition that inpaints the smooth structure and **explicitly discards the noise
  residual** rather than reconstructing it. "Simultaneous inpainting and denoising." Verified.
- **Bertalmio et al.** (IEEE TIP 2003, structure+texture inpainting): the foundational
  precedent — decompose into bounded-variation structure + texture, inpaint structure, fill
  texture separately. Maps cleanly onto bandpass-structure vs speckle.
- **Wavelet diffusion** (arXiv:2407.12538): modern deterministic-low-frequency /
  stochastic-high-frequency diffusion split — i.e. inpaint the bandpass deterministically,
  resample the noise-like high-frequency residual. Directly implementable as a loss/target split.
- **Noise2Astro** (arXiv:2209.07071): on astronomical data, self-supervised denoising
  recovers the smooth signal with 96-98% flux accuracy and (correctly) discards noise — but
  the strong results are specifically for *smooth* profiles, matching our recoverable ~0.098
  std and NOT the speckle. Honest caveat: this bounds how much win is available.

Action: this is what our `clean_smooth` target in `realify.py` already isolates. Train the
model to predict the smooth component; at inference add resampled speckle of the measured
std (0.18) + autocorr (~0.03, i.e. white) on top for a realistic-looking fill. Score the
*smooth* recovery against interp/mean-fill, and report the texture as resampled-not-recovered.

### 2. Pivot the scientific target to the complex visibility and image / power-spectrum domain
**Strongly supported; this is where the science actually lives.** Per-baseline amplitude is
the worst possible place to be judged — it's noise-dominated. The meaningful targets are
downstream, where thermal noise averages down over many visibilities.

Evidence:
- **HERA 21cm** (Pagano 2023 arXiv:2210.14927; Chen 2025 ApJ 979.191 arXiv:2411.10529):
  the EoR community embeds inpainting *inside a quadratic power-spectrum estimator* and
  propagates inpainting uncertainty into the final statistic. The scientific target is the
  power spectrum, never per-pixel recovery. Both verified, in-domain, primary.
- **VIC-DDPM** (Wang 2023 arXiv:2305.09121): conditional DDPM for radio interferometric
  reconstruction operates in the **image domain**, conditions on visibility + dirty image to
  *separate signal from noise*, and explicitly values realistic detail / faint-source
  recovery over pixel-exact fidelity.
- **TABASCAL II** (Finlay et al., A&A 701 A286, 2025; MeerKAT-era): evaluates RFI mitigation
  by **downstream image noise** (reaches RFI-free-equivalent, 10-100x lower image noise than
  AOFlagger), using a reduced-chi-squared-to-~1 stopping criterion against a known noise floor
  sigma_n — NOT pixel MAE. Directly relevant precedent for our metric choice.
- **AIRI/uSARA** (MNRAS 518.1.604): image-domain reconstruction with learned denoisers as
  plug-and-play priors; match the denoiser noise level to the measurement floor rather than
  recover individual noisy samples; evaluate with SNR/logSNR.

Caveat surfaced by verification: the Chiche/Vafaei-Sadr A&A 2023 "visibility-inpainting =
image-deconvolution equivalence" is about **incomplete uv-coverage**, a *different* inverse
problem than RFI-masked-pixel inpainting. Use it as motivation for the image-domain pivot,
not as a claim that our specific problem is dual to deconvolution.

Action: reconstruct the complex visibility V = A.e^{i phi} (phase carries the information),
write back to the MS, grid, image, and report image-domain fidelity / sensitivity. This is
the framing that makes the phase win (the durable result) the headline.

### 3. Lead with PHASE / closure quantities as the recoverable, robust target
**Supported, and matches our empirical result** (phase beats baseline ~24x; amplitude
doesn't). Phase retains coherent fringe structure that divisive norm preserves.
- **Kerrigan 2019** and the closure-quantity papers (Chael 2018 arXiv:1803.07088; Christian
  & Psaltis 2019 arXiv:1909.04681) establish phase/closure as the information-rich, gain-
  robust observables; closure phase at low SNR follows von Mises (circular) statistics, so
  model phase error with circular stats. The known, bounded information cost of closure-only
  framing (loses total flux + centroid) is documented.

Action: frame phase/complex-visibility inpainting as the contribution; report amplitude
honestly as noise-floor-limited (a genuine finding, not a failure).

### 4. Self-supervised denoise+inpaint unification (mechanism for #1 on real data)
Our Phase-2 mixed-masking is already a blind-spot self-supervised scheme. The literature
says this is exactly right and unifies denoising with inpainting:
- **Noise2Inpaint** (arXiv:2006.09450): recasts denoising as regularised inpainting on
  disjoint pixel sets — the same paradigm as Phase 2 — and folds *known noise statistics*
  into the objective. This is the concrete route to "treat the residual as noise with known
  stats" without clean ground truth.
- **VisRec** (arXiv:2403.00897): clean-GT-free self-supervised radio visibility
  reconstruction, Noise2Noise-style consistency loss, treats thermal noise as something to be
  *robust to* (injected as augmentation), reports success in the image domain. Closest
  self-supervised radio analogue, though its setting is sparse-uv VLBI not dense RFI gaps.

### 5. If you keep a single per-pixel fill, change the metric (don't change the loss alone)
- Evaluate by **distributional consistency** of the masked region (pixel-value histogram,
  masked-region mean & std), the established practice for ground-truth-free astro inpainting
  (radio-galaxy DDPM, arXiv:2601.07485, verified). Abandon FID/Inception (ImageNet-trained,
  wrong domain) and per-pixel error there.
- **PMRF** (arXiv:2410.00418): a concrete two-stage blueprint — predict the posterior mean,
  then transport it to the data distribution — i.e. exactly "recover the recoverable, then
  add realistic stochasticity." Reports both distortion and distributional metrics, including
  a no-ground-truth variant (IndRMSE) usable on real data.

## What the verification caught (don't overclaim these)
- The "posterior samples collapse to one completion" line (Cohen et al. ICLR 2024,
  arXiv:2310.16047) was **refuted** repeatedly: the paper says the posterior is *heavy-tailed*
  with genuine diverse tail mass that sampling DOES reveal, and argues for *meaningful
  diversity*, not for a distortion objective. Do not cite it for "conditional mean is
  uninformative."
- The image-deconvolution equivalence (above) is about uv-coverage, not RFI gaps.

## Concrete next steps for the pipeline
1. Re-run the speckle probe with the **full-std (0.21) structure + speckle on top** (not the
   compressed 0.10) so the smooth-recovery test isn't starved of structure to beat — settles
   whether the model beats interp/mean-fill on the *recoverable* component cleanly.
2. Switch the headline metric from amplitude MAE to: (a) smooth-component recovery vs interp,
   (b) phase angular error / complex-visibility MAE, (c) image-domain fidelity after gridding.
3. Implement decompose-then-inpaint: train on `clean_smooth`, resample speckle at inference.
4. Reframe the writeup around phase + complex visibility + image domain as the contribution,
   with the amplitude noise-floor as a quantified, literature-corroborated finding.

---

## Implementation status (updated 2026-06-23)

Steps 3 and part of 2 are now implemented. `phase1_all_decompose` checkpoint exists on ilifu
(smooth-target training, sigma=1.0). The noise-resampling at inference is live in `diffusion.py`:

- `Diffusion._estimate_noise_floor(x, keep)` — estimates per-sample, per-channel σ from the
  5×5 HP residual of the known pixels.
- `Diffusion.sample(..., noise_floor=None|'auto'|float)` — when set, adds N(0,σ²) to hole
  pixels of the final output after the denoising chain.
- `model/diagnostics/stochastic_inpaint.py` — 6-condition diagnostic (4 model conditions +
  mean-fill + interp) that reports texture ratio, MAE vs noisy target, MAE vs smooth target.
- `model/diagnostics/jobs/stochastic_inpaint.sh` — runs on `phase1_all_decompose` / `runtest`
  by default.

**Key empirical anchor (to verify on cluster):** `eta=0 +auto_noise` should give texture
ratio ≈ 1.0 with MAE vs smooth target beating mean-fill. `eta=0 no_noise` (current default
in all eval scripts) gives texture ratio ≈ 0.

---

## Additional references (deep-research sweep, 2026-06-23)

**Drozdova et al. (A&A 2024)** — conditional DDPM applied to ALMA interferometric inpainting.
Evaluated by source-detection completeness and flux accuracy, not per-pixel MAE. Closest
in-domain precedent for our evaluation framing.

**PMRF (arXiv:2410.00418)** — "Posterior Mean Rectified Flow." Two-stage blueprint matching
our decompose-then-inpaint approach exactly: (1) predict the posterior mean (recoverable
signal), (2) transport to the data distribution (resample noise). Reports both distortion and
distributional metrics, including an IndRMSE variant usable without ground truth on real data.

**Fréchet Wavelet Distance / FWD (arXiv:2312.15289, ICLR 2025)** — distributional fidelity
metric that works without a pretrained network (unlike FID) and is valid at 64px patch size.
Measures whether the inpainted region's multi-scale statistics match the surrounding data.
Directly applicable to our patches; does not require ground truth.

**L2 vs L1 diversity (Palette, Saharia et al. 2022)** — the Palette paper itself found that
L2 on the noise residual ε produces more diverse samples than L1. Our current config uses L1.
If `+auto_noise` texture ratio is still low after eval, switching to L2 is the next experiment.

**Reconciliation with `sampling-investigation.md`** — the stochastic DDPM MAE being worse than
mean-fill is the Blau & Michaeli theorem in action, not a model failure. See that doc for the
updated framing.
