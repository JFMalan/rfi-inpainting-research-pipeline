# Adversarial methodology audit (2026-06-25)

External adversarial audit of the RFI-inpainting methodology, with our response. Verdicts are
literature-grounded (HERA/LOFAR/MWA 21cm-EoR precedent). Read alongside `tiling-design-brief.md`.

## Bottom line
The literature endorses the SPECTRAL instinct but challenges the ARENA and the TOOL. All peer-reviewed
RFI-gap-recovery precedent judges inpainting in the delay/power-spectrum domain, never continuum-imaging
RMS, and the field's endorsed gap-fillers are LINEAR, statistically-tractable methods (DPSS, DAYENU, GPR)
whose signal loss is analytic. A learned diffusion model is the CHALLENGER that must beat those baselines,
not the established answer. Right corrections already made: adopt a delay-space metric. Remaining mistakes:
keeping continuum RMS/DR as the headline, and presuming the diffusion model is superior rather than
benchmarking it against the classical methods.

## Per-question verdicts
- **Q1 diffusion the right tool?** RISKY/unjustified-by-default. Pagano 2023 (arXiv:2210.14927) benchmarks
  CNN vs DPSS/CLEAN/GPR/LSSA; classical win ("DPSS and CLEAN best for intermittent narrowband; GPR and LSSA
  best for larger gaps"). Our wide persistent bands = the GPR/LSSA "larger gap" category. Chen/Kennedy 2024
  (arXiv:2411.10529) fold DPSS into a quadratic estimator precisely for statistical tractability — a
  black-box diffusion forfeits that. ACTION: add DPSS/CLEAN/GPR as mandatory baselines.
- **Q2 arena?** WRONG ARENA — switch headline to delay-space. No peer-reviewed source finds RFI-gap
  inpainting winning on continuum RMS/DR; continuum already tolerates gaps via cross-baseline uv redundancy
  (Prasad & Chengalur 2018, arXiv:1711.00128). Demote continuum to a does-no-harm check. Caveat: all
  precedent is EoR PS science; phrase as "the credible win is spectral," not "continuum is worthless."
- **Q3 full-amp target + deterministic mean + delay guard?** SOUND. Abandoning smooth/decompose is correct:
  GPR-style over-smoothing has documented signal loss when priors are misestimated (Kern & Liu 2021,
  MNRAS 501:1463) — our smooth oracle's ~6% peak suppression is exactly that. Better-principled extension
  to cite: GCR+Gibbs (Kennedy/Bull 2022, arXiv:2211.05088) — joint inpaint + PS with uncertainty. Optional
  ablation: diffusion posterior spread (N samples) vs deterministic mean.
- **Q4 recoverability ceiling on wide bands?** CONFIRMED — largely chasing info that isn't there. Pagano:
  "largest errors occur in the noise-dominated delay modes" (method-agnostic; white noise is incoherent).
  Vindicates our R²~0. Cite Chakraborty/Datta/Mazumder 2022 (arXiv:2203.04994): random flagging benign,
  contiguous flagging kills recovery. ACTION: state recoverable/irreducible split as a FINDING; stop
  optimizing against wide fully-flagged bands.
- **Q5 tiling + feathered write-back?** OPEN — no literature. Seams a priori inject high-delay power.
  ACTION: run oracle ±tiling ablation IN DELAY SPACE; tiled oracle must match non-tiled high-delay power.
  Load-bearing and currently unverified — do not assert feathering is safe without it.
- **Q6 Palette DDPM 512², cos/sin, divisive norm, freq PE?** OPEN/partial. Closest precedents: Massoud
  2024 (DDPM + mixed-masking + freq-PE) and RFI-DRUnet (arXiv:2402.13867, fully-conv, arbitrary-size).
  cos/sin-vs-complex and 2D-per-baseline-vs-uv/delay unaddressed by citation — justify by own ablations.
  NB Pagano's CNN is a per-channel U-Net, not a 512² Palette DDPM — it bounds the class, not our model.
- **Q7 self-supervised mixed-masking + referenceless eval?** SOUND (Massoud precedent). Field standard
  (Chen/Kennedy) is to quantify the inpainter's signal loss + impact on the PS statistic. ACTION: score the
  fake-hole-on-good-data proxy IN DELAY SPACE; report injected bias + variance in the high-delay statistic.
- **Q8 anything missed?** (a) TTGE gap-avoidance (Elahi/Bharadwaj 2025, MNRAS 540:2745): PS estimable from
  available channels only — sidesteps inpainting; acknowledge and argue filled-MS (pipeline-agnostic) value.
  (b) Per-baseline independence ignores array-level coherence; cross-baseline uv redundancy already handles
  continuum gaps — our single most important open question (single-baseline ceiling vs more-context).
  (c) Calibration-order and pol-collapse: flag as limitations.

## Corrections to OUR earlier framing
- **U-Paint overreach (KILL):** "U-Paint ~10^4 high-delay error vs ~10^1 classical = catastrophic" overstates
  Pagano. The paper presents the CNN as viable/transferable; classical merely edge it out on fine structure
  as noise drops. Do NOT lean on "the CNN was catastrophic." (Our own deep-research already had this at 2-1
  with caveats; treat it as softened.)
- **GOAL.md ref [12]** Luo et al. cites arXiv:2604.01531 — future-dated, almost certainly wrong ID. Verify.

## Our response (where we nuance, not concede)
- Pol-collapse is NOT untested — measured at 1.6% (repr_diag). Minor limitation, not an open assumption.
- Cross-baseline is NOT unprobed — our probe gave R²~-0.09 on a bright calibrator, uncalibrated field. Not
  settled for science fields, but already points to a real single-baseline ceiling here.
- "Diffusion isn't the established winner" bounds the class (Pagano CNN != our DDPM); response is the same:
  benchmark vs classical, don't presume superiority.

## Action list (priority)
1. Headline metric → delay/power-spectrum; demote continuum RMS/DR to does-no-harm.
2. Add DPSS + CLEAN (+ GPR) baselines the diffusion model is measured against.
3. Tiling delay-space ablation (oracle ± tiling) — load-bearing.
4. Reframe recoverable/irreducible split as a finding (cite Pagano + Chakraborty 2022).
5. Score the fake-hole proxy in delay space (bias + variance), not just amplitude RMSE.
6. Cite GCR+Gibbs and TTGE as the frontier we're not building.
7. Fix GOAL.md ref [12].
DECISION PENDING (user): switch training target from smooth/decompose to full-amplitude + delay guard.
