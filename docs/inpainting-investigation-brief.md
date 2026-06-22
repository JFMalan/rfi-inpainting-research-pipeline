# Investigation brief: why is the RFI-inpainting not working well / not smooth?

This is a NEUTRAL, factual briefing for a fresh agent to run an independent deep analysis
(e.g. via an ultracode multi-agent workflow) into why the model's inpainting of real
MeerKAT data looks poor and why fills do not blend with the surrounding data. It deliberately
separates OBSERVATIONS (facts) from PRIOR INTERPRETATIONS (hypotheses that may be wrong). Do
not assume any prior interpretation is correct — several were proposed and revised during
debugging. Re-derive from the code and data.

---

## 1. What the system is (architecture & intent)

- Goal: inpaint RFI-flagged (corrupted) regions of MeerKAT per-baseline visibility waterfalls
  so a cleaned Measurement Set can be reconstructed.
- Model: conditional DDPM, Palette-style U-Net, ~104.7M params. `predict=x0` (predicts the
  clean signal, not noise). Deterministic DDIM sampling, `eta=0`. Clamp `clip=(-2,4)`.
- Input format: 512×512 per-baseline waterfalls (time × frequency). 3 target channels:
  ch0 = amplitude (divisively normalised), ch1 = cos(phase), ch2 = sin(phase). Plus 4
  positional-encoding (PE) channels keyed to frequency. Network in_channels = 11
  (noisy x_t 3 + masked-conditioning 3 + mask 1 + PE 4), out = 3.
- Two phases: (1) supervised on SIMULATED data with clean ground truth; (2) self-supervised
  fine-tune on REAL data via "mixed masking" (fake holes placed over unflagged real pixels;
  loss computed inside the fake holes where the observed value is a known target).

## 2. Key files (read these)

Core model/training:
- `model/diffusion.py` — `Diffusion.loss` (sim/phase1), `loss_phase2` (real), `sample`,
  `_ddim_step`, `predict_x0`, `q_sample`. The train↔sample contract lives here.
- `model/data.py` — `PatchDataset` (sim), `RealDataset` (real), `build_cond`,
  `positional_encoding`, `smooth_component`, `fake_mask`, `random_mask`.
- `model/config.py` — `Config`, `phase1()`, `phase2()`; `hole_fill`, `smooth_target`,
  `smooth_sigma`, `fake_mask_mode`, `ema_decay`, in_channels.
- `model/train.py` (sim) and `model/train_real.py` (real) — training loops + `val_eval`.
- `model/unet.py` — architecture (attn_res, ch_mult).
- `model/metrics.py` — `mae`, `psnr`, `phase_error`, `complex_mae`, `tre`.

Data prep:
- `data_preparation/simulated/`: `make_bright_sky.py` / `make_random_sky.py` (sky models),
  `add_noise.py` (thermal noise into the MS), `extract_patches_sim.py` (divisive-norm +
  resize → `clean`), `inject_rfi.py` (adds RFI → `corrupted`/`mask`), `realify.py`
  (post-hoc transform: rescale + add speckle + remask, used in experiments).
- `data_preparation/real/`: `extract_ms.py` / `extract_variants.py` (MS → per-baseline h5),
  `characterise_speckle.py`, `sim_real_gap.py`, `audit_real.py`, `rfi_bands.py`.

Inference / visualisation / diagnostics (many; the relevant ones):
- `model/diagnostics/inpaint_viz.py` — produces sim/real inpaint npz (sim path is the
  one that historically looked good).
- `model/diagnostics/visualise_samples.py` — renders the npz (clean/corrupted/pred/error).
- `model/diagnostics/compare_inpaint.py` — multi-model real inpaint comparison + optional
  resampled-speckle overlay.
- `model/diagnostics/reversible_inpaint.py` — smooth→inpaint→reverse pipeline (latest, real).
- Information-theoretic diagnostics already built (use/extend them):
  `recoverability.py`, `info_ceiling.py`, `pipeline_doctor.py`, `ceiling_check.py`,
  `diagnose_model.py`, `bias_diag.py`, `sampler_sweep.py`, `speckle_probe.py`,
  `decompose_layers.py` (under simulated/visualisation), `characterise_speckle.py`.

Prior write-ups (read for history, but treat conclusions as provisional):
- `docs/phase1-science-log.md`, `docs/research-noise-dominated-inpainting.md`,
  `docs/phase2-handover.md`, `docs/sampling-investigation.md`.

## 3. Data facts (observed, on the current 512 per-baseline data)

- SIM `clean` (divnorm amplitude): mean ≈ 1.0; std ranges ~0.06 (low-noise/bright) to
  ~0.40 (diverse set) depending on run; max ~1.2 on a bright/low-noise baseline. `clean` is
  RFI-free; RFI is only in `corrupted = clean + rfi`. Verified: `clean == corrupted` on
  unflagged pixels; `clean` uncorrelated with mask position.
- REAL `data` (divnorm amplitude, file `/scratch3/users/$USER/rfi/real/variants/
  v1_upsample512.h5`): unflagged mean ≈ 1.00, median ≈ 0.99, std ≈ 0.214, p99 ≈ 1.57,
  p99.9 ≈ 2.16, max ≈ 20.7. Fraction of unflagged pixels >3 ≈ 0.02%, >5 ≈ 0.003%.
  Per-baseline unflagged max: median ≈ 5.35, worst ≈ 20.7. Real baselines are ~45–50%
  flagged, in WIDE contiguous bands (persistent MeerKAT RFI: ~930–960, 1170–1300,
  1525–1630 MHz), often flagged across all time samples in a band.
- Decomposition of real unflagged amplitude (2D Gaussian low/high split):
  - At low-pass sigma=1.0: smooth component freq lag-1 autocorr ≈ 0.92, residual ("grain")
    autocorr ≈ 0.01, smooth std ≈ 0.12, grain std ≈ 0.18. Smooth ≈ 33% of variance.
  - Larger sigma (2–4) leaves more structure in the residual (grain autocorr rises to
    0.28–0.43) — i.e. the split is sigma-dependent.
- REAL phase: per-pixel freq lag-1 autocorr ≈ 0.145 (much lower than sim's 0.4–0.99).

## 4. Quantitative results observed (held-out real, 32 test baselines, fake-hole MAE)

(Models: "decompose" = trained to predict the smooth amplitude component; "full-amp" =
trained to predict the raw amplitude. "finetune" = sim→real; "scratch" = real only.)

| model | fake-MAE | interp | mean-fill | TRE | TRE(mean-fill) |
|---|---|---|---|---|---|
| decompose finetune | 0.090 | 0.109 | 0.089 | 0.059 | 0.100 |
| decompose scratch | 0.109 | 0.109 | 0.089 | 0.134 | — |
| full-amp finetune | 0.260 | 0.207 | 0.162 | 5.34 | 5.26 |
| full-amp scratch | 0.166 | 0.209 | 0.162 | 5.18 | — |

On SIM the model performs strongly (e.g. `phase1_all`, 512, full-amp: complex-vis MAE ~0.33
vs mean-fill ~0.98; amplitude MAE ~0.13 vs ~0.35; a 256-patch bright-sky run "run2" reached
complex MAE ~0.058, amp recovery R²≈0.645). Sim fills look sharp; real fills look poor/smooth
and/or wrong-level.

## 5. The symptom to investigate (stated neutrally)

On REAL data the inpainted regions:
- (a) look much smoother / different in texture than the surrounding real data;
- (b) the user reports the fill COLOUR/LEVEL looks wrong even for small holes over
  otherwise-good data (i.e. not only the texture, but the amplitude level appears off);
- (c) some configurations produced near-constant fills (uniform bright, or uniform dark)
  inside wide bands.

The central open question: **is this a fixable bug in the setup (training/inference/
normalisation/conditioning/visualisation), or an information limit of the data (the masked
amplitude is largely unrecoverable), or a combination — and how much of each?**

## 6. Prior interpretations proposed during debugging — UNVERIFIED, may be wrong

List them so they can be tested or refuted, NOT assumed:
- "Real masked amplitude is ~67% irreducible white noise (R²≈0), so it cannot be inpainted;
  smoothness is the perception–distortion floor, not a bug." (Supported by some diagnostics;
  but note the smooth/grain split is sigma-dependent and the decompose model does beat interp.)
- "The smooth-target objective discards recoverable structure" vs "the smooth target is the
  only learnable target on real" — both were argued at different points.
- "A positional-encoding / frequency-band mismatch between sim training and real inference
  shifts the predicted level." (NOT verified.)
- "A divisive-normalisation scale difference between the sim and real prep pipelines puts the
  real amplitude on a different scale than the model expects." (NOT verified.)
- "`build_cond` hole_fill uses a hardcoded amplitude = 1.0; may not match real's conditioning
  expectation." (NOT verified as causal.)
- "Wrong colour is partly a visualisation colour-scale artifact (per-panel percentile scaling
  against real's bright tail)." (Argued then partially retracted.)
- "An EMA-decay bug (0.9999 over a short fine-tune) froze evaluation at the sim-init weights."
  (This was found and a fix applied; verify it actually took effect and whether any cited
  result predates the fix.)
- "Residual un-flagged RFI in the real targets trains the model to predict bright values."
  (Measured as sparse, ~0.02% of pixels; deemed minor — re-check.)
- "Full-time-flagged persistent bands give zero in-band context, so striping is unavoidable
  for per-baseline inpainting." (Geometric argument; test it.)

Treat ALL of the above as hypotheses with unknown truth value.

## 7. Suggested investigation directions (non-prescriptive)

A thorough audit might, among other things:
- Trace the EXACT tensor the model receives at sim-train vs real-inference (channel order,
  value scale/centering, mask convention 1=hole vs 1=keep, hole_fill value, PE construction
  and the fmin/fmax it is keyed to, clip range, ema-vs-raw weights, predict mode, steps) and
  identify any divergence between the path that works on sim and the path used on real.
- Run a ground-truthed micro-test: a small fake hole over known-good real data, and compare
  the model's fill level to the true level and to the local context (isolates "level bug" vs
  "texture/perception"). A partial version of this exists; extend it across many baselines and
  hole sizes/positions, and across frequency (to expose any PE/band dependence).
- Quantify recoverability of the masked content directly (context→hole R², autocorrelation,
  per-frequency) on REAL data with the current 512 format — the existing `info_ceiling.py` /
  `recoverability.py` do versions of this; confirm/refresh the numbers and check whether
  conclusions are sigma/scale dependent.
- Separate, per pixel in the hole: the recoverable component the model gets right vs the
  residual it cannot, to attribute the visible error between "model" and "physics".
- Verify normalisation parity: compare the divisor/normalisation applied to sim `clean` vs
  real `data` and whether the model sees the same scale.
- Confirm whether "wrong colour" is a real level error (numbers) or a display artifact
  (shared vs per-panel colour scaling); fix the diagnostic if it is the latter.
- Distinguish the wide-fully-flagged-band regime (no in-band context) from partially-flagged
  holes (context available) when judging fill quality.

## 8. Environment / how to run (the agent cannot reach the cluster directly)

- All compute is on the **ilifu** SLURM cluster. The agent has NO direct cluster access — it
  can read/edit code in the repo, but the USER runs commands. Edits reach the cluster only via
  local git commit+push → `git pull` on ilifu. Always pull before sbatch.
- Containers: `/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif` (PyTorch, GPU),
  `/idia/software/containers/ASTRO-PY3.10.sif` (h5py/numpy/scipy/matplotlib, CPU analysis).
- GPU jobs: `--partition=GPU --gres=gpu:1 --constraint=A100|A40|V100`; for short jobs add
  `--qos=qos-interactive` (≤4 CPU, ≤28GB) to skip the queue. Bind driver libs (see existing
  job scripts under `*/jobs/` for the `--bind libcuda/libnvidia-ml` pattern; `--nv` alone
  fails on these nodes).
- Data: real = `/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5`;
  sim = `/scratch3/users/$USER/rfi/simulated/run[1-9]/dataset.h5`.
- Checkpoints: `/idia/users/$USER/rfi/runs/` — `phase1_all/best.pt` (512 full-amp sim),
  `phase1_all_decompose/best.pt` (512 smooth-target sim),
  `phase2_decompose/v1_upsample512_{finetune,scratch}/best.pt` (real, smooth-target),
  `phase2_decompose_fullamp/...` (real, full-amp). Each ckpt stores its `cfg`.
- Logging convention for any new diagnostic/cluster script: print device (cuda/cpu + GPU
  name), key milestones, and periodic progress with elapsed time/rate (flushed).

## 9. Output wanted from the analysis

A ranked, evidence-backed diagnosis of why the real inpainting looks poor / not smooth,
distinguishing (with code citations and/or measured numbers): genuine setup bugs (fixable)
vs information-theoretic limits (not fixable) vs visualisation artifacts. For each claimed
cause: the evidence, a decisive test to confirm it, and the concrete fix if applicable.
State plainly what is ruled out so it is not chased further.
