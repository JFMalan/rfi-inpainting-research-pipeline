# Results Summary — RFI Inpainting (conditional DDPM)

Consolidated results as of 2026-07-21. All simulated numbers are against noise-free clean
truth on the held-out `runtest` split; real numbers use the fake-hole delay metric (the only
ground-truthed real metric).

---

## 1. Component ladder (Massoud recipe → our recipe), simulated

What each design component contributes, built one rung at a time on the fixed run1-3 subset
(same data, same 30-epoch budget per rung). Evaluated on `runtest`.

| Rung | Adds | complex_mae | amp_mae | PSNR (dB) | phase_err |
|------|------|-------------|---------|-----------|-----------|
| R0 | Massoud base (amp-only, raw amps, **no div-norm**, noisy target) | 4.902 | 4.902 | 13.56 | — |
| R1 | **+ divisive normalisation** | 0.191 | 0.191 | 23.14 | — |
| R2 | + cos/sin phase (complex reconstruction) | 0.568 | 0.282 | 20.09 | 0.631 |
| R3 | + noise-free (clean) target | 0.387 | 0.205 | 22.37 | 0.411 |

Reading:
- **Divisive normalisation is the decisive component.** R0→R1 drops complex_mae 4.90 → 0.19
  (26x) and lifts PSNR 13.6 → 23.1 dB. Without it the model fails outright.
- Adding **phase** (R2) raises complex_mae because the metric now scores a harder 3-channel
  task; amp_mae 0.282 and PSNR 20.1 dB show the amplitude channel stays strong.
- The **noise-free target** (R3) improves every metric over R2: complex_mae 0.568 → 0.387
  (-32%), amp_mae 0.282 → 0.205 (-27%), PSNR 20.1 → 22.4 dB, phase_err 0.631 → 0.411 (-35%).
  R3 recovers nearly all of the amplitude cost of adding phase while also reconstructing phase.
- All rungs beat mean-fill (amp_mae ~0.31).

Caveat: R2 and R3 (3-channel, slower/epoch) were walltime-limited (best.pt at ~epoch 16-17 of
the 30-epoch budget), so their numbers are conservative lower bounds; R0/R1 are amplitude-only.

### 1a. R4 — sampling / write-back (inference-only on R3)

R4 adds the sampling `noise_floor` knob on R3 (no retrain), scored in delay space vs the
noise-free clean truth on runtest (300 tiles).

| Method | wlogP-RMSE | hi-ratio |
|--------|-----------|----------|
| model nf=none | 0.0277 | 1.04 |
| model nf=auto | 0.0413 | 1.07 |
| model nf=0.3 | 0.0536 | 1.12 |
| model nf=0.5 | 0.0846 | 1.28 |
| DPSS | 0.0955 | 0.86 |
| flagged | 0.1251 | 0.84 |
| GPR | 0.525 | 1.72 |

Model recovers the delay spectrum 3.4x better than DPSS (4.5x vs flagging), near-perfect
hi-ratio (1.04). Against the noise-free target **nf=none is optimal** — adding noise_floor
over-recovers (hi-ratio 1.12 -> 1.28). Complements the real-data result where nf=0.5 was needed
to match real grain: nf=none for the noise-free science target, nf>0 only for real-data texture.

### 1b. Representation ablation — amp+cos/sin vs real+imag

Controlled test (no prior published run): an otherwise-identical real+imaginary (2-channel)
model on the R3 recipe/budget, scored on the same runtest 512-subset.

| Representation | complex_mae | amp_mae | PSNR | phase_err | epochs |
|----------------|-------------|---------|------|-----------|--------|
| amp + cos/sin (3ch) — R3 | 0.387 | 0.205 | 22.37 | 0.411 | ~17 |
| real + imaginary (2ch) | 0.556 | 0.291 | 19.32 | 0.628 | 18 |

**amp+cos/sin wins decisively on every metric.** Real/imag does not reduce the amplitude
penalty — it enlarges it (0.291 vs 0.205) — despite training a full epoch longer. Validates the
pipeline's amp+cos/sin choice: the bounded cos/sin target is better-conditioned than real/imag,
whose wider dynamic range and amplitude-entangled channels the L1 diffusion loss fits less
uniformly.

---

## 2. Simulated ablations — in-paint vs flagging

Realistic RFI (persistent bands + bursty narrowband + broadband bursts + sweeps) unless noted.
Continuum = image RMSE vs clean (lower better); delay = wlogP-RMSE vs clean (hi-ratio in
parens, 1.0 = perfect).

### Sky brightness (per-visibility SNR), fixed 37% flagged
| SNR | Continuum adv. | Delay: model | Delay: flagged | Delay adv. |
|-----|----------------|--------------|----------------|------------|
| ~90 | 2.5x | 0.0092 (0.996) | 0.283 (0.829) | 31x |
| ~30 | 2.4x | 0.0105 (0.995) | 0.286 (0.825) | 27x |
| ~9  | 1.9x | 0.0217 (0.985) | 0.289 (0.803) | 13x |
| ~3 (real regime) | 1.9x | 0.0488 (0.948) | 0.283 (0.748) | 6x |

At SNR~3 the in-paint image preserves source photometry (peak 0.0775 vs clean 0.0774; flagged
0.0749; mean-fill collapses to 0.031). Low-SNR continuum CIs are tight and non-overlapping.

### Flagged fraction, realistic RFI at full brightness
| Fraction | Continuum adv. | Delay: model | Delay: flagged | Delay adv. |
|----------|----------------|--------------|----------------|------------|
| 15% | 2.2x | 0.0059 (0.998) | 0.162 (0.858) | 27x |
| 30% | 2.4x | 0.0073 (0.998) | 0.222 (0.856) | 30x |
| 45% | 2.6x | 0.0109 (0.999) | 0.356 (0.798) | 33x |
| 60% | 2.4x | 0.0164 (1.004) | 0.545 (0.677) | 33x |

The delay-space margin widens with fraction (flagging's error triples as gaps grow; the model
stays near-perfect). Continuum lead flat at ~2.3x.

### Band-width sweep (uniform stripes, 30% flagged) — reference upper bound
Continuum advantage: 12.6x (1ch) → 6.0x (4ch) → 2.4x (32ch) → 1.5x (64ch). Width is the weak,
saturating axis; fraction and SNR are the variables that matter.

---

## 3. Real MeerKAT (1570802018, target J2018-5539) — the headline for observers

No clean truth exists on real data, so the ground-truthed metric is fake-hole delay recovery:
punch synthetic holes over known-good real pixels, in-paint, score the delay spectrum against
the untouched data. 400 baseline-tiles, 900.6-1649.6 MHz.

### Fake-hole delay recovery — wlogP-RMSE (lower better), advantage = method / model
| Hole regime | model | DPSS | GPR | zero-fill |
|-------------|-------|------|-----|-----------|
| mixed 10-25% | 5.2e-5 | 2.21e-4 (**4.3x**) | 9.67e-4 (18.7x) | 4.75e-4 (9.2x) |
| high 40-55%  | 1.0e-4 | 1.52e-4 (**1.5x**) | 1.51e-3 (15.1x) | 6.38e-4 (6.4x) |

- The model wins outright at every regime; ordering model < DPSS < zero-fill < GPR is stable;
  model-vs-DPSS bootstrap gap significant (P < 0.001).
- Honest nuance: the metric is power-weighted, so the decisive margin is around the bright
  foreground peak where the model tracks truth almost exactly. In the far delay tails the model
  and DPSS are competitive; at high flagged fraction DPSS nearly catches up (1.5x).

### Fine-tune vs from-scratch (real, phase-2)
Validation fake-hole delay: fine-tuned (sim prior) **0.00313** vs from-scratch **0.00699** (~2.2x
better). The simulation prior transfers.

### Real continuum imaging (no-truth diagnostic, descriptive only)
Flagged off-source RMS 2.07e-5 vs in-paint 2.73e-5. Off-source RMS measures the noise floor,
not correctness — the in-paint fills flagged visibilities with structured signal (raising
residuals near the bright core while restoring uv-coverage). The ground-truthed verdict is the
delay metric above.

---

## 4. Job provenance
- Ladder: massoud_r0-r3, `evaluate_sim` on runtest. R2 train 309707, R3 train 309709 (TIMEOUT at
  36h; best.pt saved). Evals 309708/309710 COMPLETED.
- R4: sim_delay_eval on R3 best.pt (job 372671, 300 tiles, noise_floor sweep) — COMPLETED.
- Representation ablation: real/imag train (job 372668, TIMEOUT 36h @ epoch 18), runtest eval
  (job 391359, 512-subset matched to R3) — COMPLETED. Code: `vis_repr` flag in config/data/
  metrics/train/evaluate.
- Real HERA/Pagano benchmark: `pagano_real_eval.py` (job 360600, 400 tiles, shifted-flag holes,
  full classical set incl. CLEAN+LSSA) — COMPLETED.
- Real fake-hole delay + fine-tune-vs-scratch (val_delay 0.00313 vs 0.00699) — from phase-2
  training logs (jobs 290550/290551).
