# Phase 2 / sim→real transfer — handover (2026-06-22)

State of the RFI-inpainting investigation after the sim-vs-real gap analysis, the decompose
approach, the audit, and the reverse-smoothing pipeline. Read alongside
[phase1-science-log.md](phase1-science-log.md) and [research-noise-dominated-inpainting.md](research-noise-dominated-inpainting.md).

## The one-paragraph summary
The model inpaints amplitude+phase **excellently when the data has recoverable structure**
(bright/structured sim) and **cannot recover amplitude when it doesn't** (real MeerKAT
blank-field amplitude is ~67% irreducible white noise: context→hole R²≈0, freq lag-1
autocorr≈0.03). This is an honest SNR limit, not a model failure — proven repeatedly and
now with post-bug-fix ablations. **Phase is recoverable on sim but weak on real** (real phase
freq-autocorr ≈0.145). The defensible contributions are: (1) the pipeline genuinely inpaints
structured data; (2) a quantified, ablation-backed characterisation of *what is and isn't
recoverable* in real visibility amplitude; (3) a regime-dependent target choice
(full-amplitude for high-SNR sim, smooth-component for noise-dominated real); (4) a
reversible smooth→inpaint→reverse pipeline that produces realistic-looking real fills.

## Key numbers (held-out, real v1_upsample512, 32 test baselines, fake-hole MAE)
| model | fake-MAE | interp | mean-fill | TRE | TRE_mf | verdict |
|---|---|---|---|---|---|---|
| decompose finetune (sim→real, smooth target) | 0.090 | 0.109 | 0.089 | 0.059 | 0.100 | beats interp; **TRE 40% < mean-fill** |
| decompose scratch (real only) | 0.109 | 0.109 | 0.089 | 0.134 | — | only ties interp → **sim prior helps** |
| full-amp finetune (sim→real, raw amp) | 0.260 | 0.207 | 0.162 | 5.34 | 5.26 | loses to both (degenerate bright fill) |
| full-amp scratch | 0.166 | 0.209 | 0.162 | 5.18 | — | beats interp, ties mean-fill |

Interpretation: the **decompose finetune is the best real model** — beats interp on amplitude
and beats mean-fill on TRE by a clear margin; the from-scratch ablation shows the **sim prior
is necessary** (real-only just ties interp). Full-amplitude on real is degenerate (the noise
ceiling is real — confirmed after the EMA fix below).

## Checkpoints (/idia/users/$USER/rfi/runs/)
| Path | Format | Target | Notes |
|---|---|---|---|
| `phase1_all/best.pt` | 512 | full-amp | **best 512 sim model**, diverse 5-run set, complex MAE 0.33 (3× mean-fill), sharp sim fills. The "good inpainter" on sim. |
| `phase1_all_decompose/best.pt` | 512 | smooth | 40-epoch smooth-target sim prior (used for the decompose finetune) |
| `phase1_all_decompose_80ep/best.pt` | 512 | smooth | longer sim run (was launched 2026-06-21; check if finished) |
| `phase2_decompose/v1_upsample512_finetune/best.pt` | 512 | smooth | **best real model** (0.090, beats interp) |
| `phase2_decompose/v1_upsample512_scratch/best.pt` | 512 | smooth | real-only ablation |
| `phase2_decompose_fullamp/...` | 512 | full-amp | the failed-on-real control |
NOTE: the legendary `run2` (complex MAE 0.058, R²=0.645) was **256-patch bright-sky**, NOT 512
— it does not load into the current 512 UNet. Best 512 sim model is `phase1_all`.

## CRITICAL bugs found & fixed this session (the audit, 2026-06-21)
The first real-data run looked like a hard negative ("can't beat mean-fill"). An independent
audit found the negative was **untrustworthy** — three issues made the test unwinnable
regardless of model quality:
1. **EMA decay 0.9999 froze the eval at the sim init.** Half-life ~6931 steps vs an ~800-step
   fine-tune → every `best.pt`/val was ≈ the sim checkpoint, not the fine-tuned model. FIX:
   `train_real.py` now auto-scales EMA to run length (or `--ema-decay`, job uses 0.999). This
   alone flipped `beats_mf` false→true.
2. **Smooth target == the interp baseline by construction.** `smooth_component` interp-filled
   the hole across freq then blurred; fake holes were full-time freq stripes → target inside
   the hole ≈ what interp/mean-fill produce → unwinnable. FIX: **2D-blob fake holes**
   (`fake_mask_mode='2d'` in data.py) so recovery needs cross-time+freq context.
3. **Metric rigged on a flat target + TRE meaningless.** MAE-vs-mean-fill on a std~0.08 smooth
   target is unwinnable (Blau-Michaeli); TRE fidelity term ≈0 for the model (kept region
   pinned). FIX: primary bar is now **vs interp**, plus a fill-std/true-std metric in
   `eval_real.py`.
Confirmed NOT broken: no conditioning leak; Palette contract correct; not undertrained-by-iters.

## Decompose-then-inpaint (the working real approach)
- `smooth_target` flag in `data.py`/`config.py`/`train.py`/`train_real.py`: model predicts the
  **recoverable smooth bandpass**, not the noisy amplitude. 2D Gaussian low-pass.
- **sigma=1.0** chosen by a sweep on real (`data_preparation/real/jobs/sigma_sweep_real.sh`):
  cleanly splits structure (smooth freq-autocorr ~0.92) from white noise (grain autocorr ~0.01).
  Recoverable smooth component ≈ **33% of real amplitude variance** (the rest is irreducible).
- On sim the decompose model LOOKS bad because it's judged vs raw textured clean (a comparison
  mismatch, not a bad model) — for SIM demos use the full-amp `phase1_all`.

## Reverse-smoothing pipeline (smooth → inpaint → reverse) — the realistic-fill route
`model/diagnostics/reversible_inpaint.py` + `jobs/reversible_inpaint.sh`. Per baseline ×
method (gaussian/median/wavelet):
1. **decompose** observed amp → `low` (hole-aware normalized-convolution Gaussian, built only
   from unflagged pixels) + `high = data − low` (exact on observed pixels → mathematically
   reversible, recon error ~1e-7).
2. **inpaint** `low` with the chosen model.
3. **reverse**: outside the RFI band keep the untouched observation (true reverse); inside the
   band synthesise `low_filled + resampled white noise at local std` (the true high is
   unrecoverable — band is fully corrupt) and **level-match** the in-band fill to the NC
   reference so there is no colour step at the band edge.
Latest state: ran `phase1_all` through this on real; v1 had vertical-streak `low` artifacts +
band-edge colour steps; **v2 fixes both** (hole-aware NC low-pass + per-row level-match) — was
the last thing committed; the v2 figure had not been reviewed yet at handover.

## What is definitively settled (do not re-litigate)
- Real MeerKAT blank-field **amplitude** in the hole is ~67% irreducible white noise →
  not inpaintable by any model (R²≈0). Full-amp on real is degenerate even post-EMA-fix.
- The decompose/smooth-target approach **works on real** (beats interp, beats from-scratch) —
  the sim prior helps.
- The data pipeline is **correct**: sim `clean` is RFI-free (max 1.17 vs corrupted 5.96),
  untouched outside the mask, uncorrelated with mask position. The "vertical stripes in clean"
  scare was a viz bug (`inpaint_viz.py` was showing `smooth_component(clean)` not raw clean) —
  fixed, shows raw clean now.
- Residual RFI in "unflagged" real pixels is real but **sparse** (0.02% of pixels >3, per-
  baseline max ~5–20) — minor contaminant, not the cause of the degenerate full-amp fills.

## Open threads / next steps
1. **Review the v2 reverse-pipeline figure** (`phase1all_reversible_v2.png`) — confirm the
   streaks and band-edge steps are gone. If good, this is the realistic real-fill deliverable.
2. **80-epoch sim run** (`phase1_all_decompose_80ep`) — if finished, optionally re-run the real
   decompose finetune from it (`INIT=...80ep/best.pt`) to see if a stronger prior lifts 0.090.
   Expectation: small, since real recoverable signal is small.
3. **Clip residual RFI** in real targets (winsorize unflagged at ~p99.9 ≈ 2.2) as hygiene; the
   user chose "clip + re-test smooth" — not yet run.
4. **Phase recoverability on real** — real phase autocorr is only 0.145 (vs 0.4–0.99 sim). The
   "phase is the durable contribution" story rests on sim; verify it holds on real before
   leaning the thesis on it.
5. **Writeup framing**: regime-dependent target (full-amp sim / smooth real) + honest SNR
   ceiling + sim-prior-helps ablation + reversible realistic-fill pipeline.

## Workflow reality (important)
- All runs are on **ilifu/SLURM**; the assistant has **no cluster access** — the user runs
  every command. Edits reach the cluster only via local commit+push → `git pull` on ilifu.
  Always `git pull` before `sbatch`; jobs read the script at run time.
- GPU jobs: pin/queue on **gpu-006** (A40). Quick viz/eval jobs use
  `--qos=qos-interactive` + `--constraint=A100|A40|V100` to jump the queue (mem ≤28GB).
  Containers: `ASTRO-GPU-PyTorch-2026-01-28.sif` (torch), `ASTRO-PY3.10.sif` (analysis).
- Diagnostic scripts added this session: `data_preparation/simulated/realify.py`,
  `model/diagnostics/speckle_probe.py`, `decompose_layers.py`, `compare_inpaint.py`,
  `reversible_inpaint.py`, `vis_speckle.py`; jobs under `*/jobs/`.
