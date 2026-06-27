# Methodology audit — verdict, actions, and citations

Two independent research-agent audits (2026-06-24 strategy audit, 2026-06-27 adversarial audit) reached the
same core verdict. This doc tracks the verdict, what we've already changed, what's left, and the citations to
integrate into the thesis. Citations below are **audit-sourced — verify each against the actual paper before
citing** (the 2026-06-27 audit already self-corrected one of its own claims; treat its references as leads,
not facts).

## TL;DR verdict

The literature endorses the **spectral (delay / power-spectrum) domain** as the arena where RFI-gap
inpainting is credited, and treats **linear, statistically-tractable gap-fillers (DPSS, DAYENU, GPR)** as the
established methods. A learned diffusion model is the **challenger that must beat those baselines**, not the
default answer. Our recent corrections (drop the smooth/decompose target, add a delay-space metric) are right.
The remaining mistake was keeping **continuum imaging RMS/DR as the headline** and presuming the diffusion
model is superior rather than benchmarking it against the classical methods.

## The reframe

- **Headline metric = delay / power-spectrum** (FFT along frequency). Continuum RMS / dynamic range is demoted
  to a "does no harm" sanity check. This is where our oracle already beats flagging, and the only arena the
  field credits for inpainting.
- **Thesis question** (sharper, honest, publishable either way):
  *Can a learned generative prior recover the ~⅓ coherent in-gap structure better than the linear filters the
  field endorses (DPSS / GPR), at the cost of statistical tractability?*
  - Beats DPSS/GPR in delay space on that ⅓ → a genuine contribution.
  - Only ties while being a black box → classical methods remain preferable. Still a valid thesis outcome.

## Per-question verdict (condensed)

| # | Question | Verdict |
|---|----------|---------|
| 1 | Is diffusion the right tool? | RISKY / unproven-by-default. Must benchmark vs DPSS/CLEAN/GPR. |
| 2 | Continuum or spectral arena? | WRONG arena — switch headline to delay space. |
| 3 | Full-amp target + deterministic mean + delay guard? | SOUND. Smooth/decompose correctly abandoned (GPR-style over-smoothing signal loss). |
| 4 | Recoverability ceiling on wide bands? | CONFIRMED — largely irreducible; frame as a finding, not a model failure. |
| 5 | Tiling + feathered write-back? | Largely settled empirically (level-0 oracle reproduced clean delay spectrum exactly, ratio 1.000). |
| 6 | Architecture/representation? | Open — justify by our own ablations, don't overclaim literature support. |
| 7 | Self-sup mixed-masking + referenceless eval? | SOUND; sharpen by scoring the fake-hole proxy in delay space. |
| 8 | Anything missed? | TTGE (gap-avoidance) competing paradigm; per-baseline vs array-level context; pol/cal untested. |

## Actions

DONE
- Native-resolution frequency tiling + feathered write-back (kills the 898→512 amplitude undersample).
- Phase stored wrap-safe (atan2 of resized sin/cos).
- Delay-space metric (`evaluation/delay_spectrum.py`).
- Pivoted off the smooth/decompose target to full-amplitude (smooth-fill oracle LOST to flagging: RMSE
  2.85e-4 vs flagged 2.28e-4, peak suppressed ~6%; full-fill oracle WINS at 1.71e-4).
- **DPSS classical baseline in delay space** (`evaluation/classical_fill.py`, `delay_spectrum.py --dpss`,
  `image_eval.sh DPSS=1`). Ridge-regularised, delay-half-width-limited; the method the model must beat.
- Delay space promoted to the headline; verdict line now asks "does the model beat DPSS / flagging".

TO RUN
- Calibrate DPSS vs the oracle ceilings in delay space on sim run1 (clean / flagged / DPSS / full-fill).
- After the full-amp model trains: image with `SMOOTH=0 NOISE_FLOOR=none` + `WEIGHT_FRAC` sweep, delay-space
  headline, against DPSS.
- (Optional) add CLEAN and GPR baselines for completeness.

WRITEUP / POSITIONING
- Frame the recoverable (~⅓) / irreducible (~⅔ thermal grain) split as a headline scientific finding.
- Acknowledge TTGE / gap-avoidance as the competing paradigm; argue our contribution is a **filled MS usable
  by any downstream pipeline**, not a power-spectrum estimate.
- Cite GCR+Gibbs as the principled uncertainty-aware extension we are not building.
- **Verify the U-Paint framing against Pagano directly** before any claim — the audit says the CNN is presented
  as viable (classical merely edges it out on fine structure as noise drops), which contradicts the earlier
  "10⁴ catastrophe" framing in our brief. One is wrong; it's a one-paper read.
- Limitations section: polarization collapse (one V per pol) and calibration-order interactions are untested.

## Citations to integrate (verify each)

- Pagano et al. 2023, MNRAS 520:5552 / arXiv:2210.14927 — benchmarks CNN vs DPSS/CLEAN/GPR/LSSA; DPSS/CLEAN
  best for intermittent narrowband, GPR/LSSA best for larger gaps; noise-dominated delay modes unrecoverable
  by any method (the information-theoretic ceiling). **Read directly to settle the U-Paint framing.**
- Chen & Kennedy 2024, ApJ 979:191 / arXiv:2411.10529 — DPSS folded into a quadratic estimator; missing data
  rings bright foreground modes in Fourier space; argues for statistically-tractable fillers.
- Kern & Liu 2021, MNRAS 501:1463 — GPR over-smoothing / signal loss when the prior is misestimated (explains
  our smooth-fill peak suppression).
- Kennedy & Bull 2022, arXiv:2211.05088 — Gaussian Constrained Realisations + Gibbs: principled
  inpaint-with-uncertainty (the frontier we're not building).
- Chakraborty, Datta & Mazumder 2022, arXiv:2203.04994 — random flagging benign, contiguous flagging is the
  hard case (formalises why our wide persistent bands are hard).
- Prasad & Chengalur 2018, arXiv:1711.00128 — cross-baseline uv-redundancy already handles gaps in continuum
  (why nobody reports continuum inpainting wins).
- Elahi & Bharadwaj 2025, MNRAS 540:2745, DOI 10.1093/mnras/staf896 — TTGE: estimate the power spectrum from
  available channels without filling gaps (the competing paradigm).
- Massoud et al. 2024 — DDPM + mixed-masking on fully-corrupted data (our direct training-approach precedent).
- RFI-DRUnet, arXiv:2402.13867 — fully-convolutional arbitrary-size net (the named alternative to tiling).
- Saharia et al. 2022 (Palette) — the conditional-diffusion architecture basis.

## Open empirical questions (settle by our own ablations, not citation)

- **Per-baseline vs array-level context** — would cross-baseline uv-information recover more than single-baseline
  on wide bands? Our cross-baseline probe gave R²≈0 (single-baseline ceiling), but on a bright uncalibrated
  calibrator field; not a clean verdict. The audit calls this the single most important open question.
- Representation: cos/sin vs complex-valued; 2D-per-baseline vs uv/delay-domain operation.
- DPSS hyperparameters (delay half-width, ridge) — sweep; ideally set the half-width by the per-baseline horizon.
