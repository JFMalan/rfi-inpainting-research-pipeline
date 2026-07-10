# final.md audit — requirement by requirement

Status as of 2026-07-10, after the restructure (commits `d9545a9`..HEAD on `final`).
MET = implemented and locally verified; CLUSTER = implemented, needs the listed cluster
run to confirm; OPEN = not done, with the reason.

## Ground rules

| Req | Status | Notes |
|-----|--------|-------|
| Reuse, don't rewrite | MET | only the sanctioned new code written; production logic untouched except verified bug fixes |
| Verify before changing | MET | three final.md/lore errors found and corrected: delay fill is noise_floor=0.5 not "matched" (user decision, 2026-07-04 sweep); Massoud loss is two L2 terms not "L1+L2" (read the paper); GOAL.md ref [12] arXiv:2604.01531 is REAL (audit's "bogus ID" claim wrong; year fixed 2024→2026) |
| Nothing lost | MET | tag `pre-restructure`; 59 scripts to archive/ with era map; nothing deleted |
| Cluster reality | MET | known-good numbers kept (train 144h, write-back 128GB, imaging 64GB/4h); ilifu specifics verified against .claude/docs |
| Code style + logging | MET | new scripts follow both rules |

## Keep vs archive

MET — `docs/restructure-inventory.md` (141 files), moves in commits 48a6211/0ec7926/49b9794.
Gates live in `tests/`, not archive.

## Target architecture

| Req | Status |
|-----|--------|
| Two-level YAML (telescope/experiment) | MET — configs/ + orchestrator/configlib (PyYAML-free fallback for the login node, cross-checked by tests/config_check.py) |
| HERA stub, documented untested, uvh5 noted | MET — cited values, derivation caveats flagged in-file |
| MeerKLASS dropped | MET — never added |
| No speculative abstraction | MET — flat env-var bridge, no plugin layers |

## Simulated data

| Req | Status |
|-----|--------|
| 10 train + 1 test run, deterministic, regenerable by one command | MET — `run_pipeline.sh configs/experiment/final.yaml`; seeds = seed_base + run |
| Random sky per run, normal flux range (not 40x bright) | MET — GEN_RANDOM_SKY=1, sky seeded per run; bright sky archived and defaults flipped |
| Noise uniform 0.7–1.0x SEFD, test at 1.0 | MET — deterministic per-run draw in resolve_config |
| RFI diversity, mean flag frac ≤ 50% | MET — per-run draws: flag frac 0.15–0.50, persist 0.3–0.8, scale 5–50 |
| ≥2 non-overlapping 512-bin time windows | MET — tiling.time_window_starts + inject_rfi double loop (SYNTHESIS 2.4h → 2 windows); freq-tile overlap deliberately kept |
| Budget < 100k samples | MET — 10 × 2016 × 2 × 2 ≈ 80,640 |
| Clean amp AND phase targets, native pairing, shared noisy divisor, no retrofit script | MET — CLEAN_DATA snapshot in add_noise; amp_target/phase_target through extraction and tiling; retrofit make_paired_dataset archived |
| smooth_component never a target | MET — legacy flag only, off everywhere; mutually exclusive with clean_target |

Bonus fix found by the smoke run: pre-restructure thermal noise was FLAT at the 880-MHz
SEFD (casatools CHAN_FREQ shape bug); now frequency-shaped, residual applied to both
noisy columns. Past datasets: internally consistent but flat-noise — say so in the report.

## Training

| Req | Status |
|-----|--------|
| Phase 1 on 10 runs, mean-fill val baseline kept | MET |
| Val-eval step count configurable | MET — --val-eval-steps / config val_eval_steps |
| Phase 2 both configs always | MET — orchestrator submits finetune + scratch |
| 40k cap, >85% flagged excluded, split by baseline | MET — max_samples→--max-patches; extraction already excludes >50% flagged units (stricter than 85%); baseline-grouped splits unchanged |
| Real recipe unchanged (noisy target) | MET — loss_phase2 untouched |
| Real dataset re-extracted fresh | CLUSTER — extract_real stage on the J2018 corr MS (pre_flagged=true per user); needs the run |
| Test-sample panels, full + selective | MET — panels_real_{full,selective} stages |

## Evaluation suite

| Req | Status |
|-----|--------|
| 5 variants in both arenas | MET — flagged/DPSS/GPR/inpaint-all/selective; GPR continuum via new gpr_fill_write.py (GPRFILL=1) |
| Per-arena write-back mode automatic | MET — config noise_floor: continuum none, delay 0.5 (user decision over final.md's "matched"; ablation still sweeps) |
| Metrics + bootstrap CIs everywhere | MET — delay eval already bootstraps; compare_images.py gained block-bootstrap CI95 on RMSE-vs-clean |

## Ablations

| # | Status |
|---|--------|
| 1 Massoud ladder | MET (configs massoud_r0–r3, shared run1-3 subset, 30 ep) — R0 pinned to the actual paper (loss=l2, no normalization → raw amps); R4 is inference-only: run fakehole_delay/image_eval with noise_floor/matched-grain flags against the R3 checkpoint — OPEN as a config, runnable by hand |
| 2 Clean vs noisy target | MET — ablation_noisy_target.yaml (own MS column, same datasets) |
| 3 Weighted imaging sweep | OPEN as one-command config — machinery kept (set_holes_weight, downweight_delay_queue.sh) and runnable stage-by-stage |
| 4 Sampling techniques | MET — fakehole_delay.sh env sweeps (NOISE_FLOORS/eta/repaint_u/ensemble/steps) |
| 5 Native tiling vs downsample-512 | OPEN as config — extract_variants.py + compare_variants.sh kept and runnable |
| 6 Flag-fraction / width sweep | MET — inject_width.sh + rfi_width_sweep.sh + plot_width_sweep.py kept |
| 7 Noise generalisation | MET — lecturer_experiments.sh kept (now uses the normal sky by default) |

## Figures pipeline

MET — figures/ area, one generator per figure, `figures/README.md` maps figure → script →
inputs → producing stage; new train_curves.py + massoud_ladder.py; no GAP entries.

## Master orchestrator

MET — run_pipeline.sh + orchestrator/{submit,status}.py: 28-stage afterok DAG, state file,
resume (COMPLETED skip / RUNNING reuse / failed resubmit), --dry-run/--only/--force,
per-stage logs under logs/, status table with log paths. Stages individually runnable via
resolve_config + sbatch.

## Reproducibility & housekeeping

| Req | Status |
|-----|--------|
| Fixed seeds end-to-end, documented in config | MET |
| Checkpoints on /idia + model inventory | MET — README table |
| Containers documented per stage | MET — README table (ASTRO-GPU.simg trap called out) |
| README rewritten | MET |
| CLAUDE.md + GOAL.md updated, .claude/docs untouched | MET |
| Validation gates runnable + documented | CLUSTER — tests/ + README section; repr_diag/oracle level-0 must re-run on the clean-target extraction before the production run |

## New code allowed — all four delivered

GPR MS write-back; clean-phase target plumbing; YAML config layer + orchestrator +
figures scripts; Massoud R0 config (config/flags on the existing trainer, no new trainer).

## Landmines — all guarded

noise_scale=0 DATA→CORRECTED copy kept; smooth_component never a target; EMA verify
behaviour untouched; RESET_COL default on; GPR constant-mean; divisor = 64-bin noisy
running mean, shared by clean targets; val-eval steps configurable; scratch regenerable.

## Open items (explicit)

1. Cluster verification round 2: clean-target smoke sim run, repr_diag + oracle level-0
   on the new h5, full-pipeline --dry-run on ilifu, then the production submission.
2. Massoud R4 rung and ablations #3/#5 have kept machinery but no one-command experiment
   config; run stage-by-stage or add configs when scheduling them.
3. Massoud loss faithfulness: rungs use plain L2 in our hole-only Palette loss; the
   paper's α-weighted known-region term is not replicated (documented deviation — their
   mixed-masking is captured by rand_mask, which is the component that matters).
4. Report must note the pre-2026-07-10 flat-noise characteristics of any old-figure data.
