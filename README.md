# RFI inpainting for MeerKAT visibilities with conditional diffusion

A conditional DDPM (Palette-style) that inpaints RFI-flagged regions of MeerKAT L-band
spectrograms directly in the visibility domain: amplitude and phase (as cos/sin) are
reconstructed per baseline, rebuilt into complex visibilities, and written back into the
Measurement Set, so the fill can be imaged and delay-transformed like real data. Flagging
is the baseline to beat; classical DPSS and GPR gap-fills are benchmarked alongside.

Everything here runs on the ilifu (IDIA/SARAO) HPC cluster via SLURM and Singularity
containers. Nothing assumes a local GPU or a writable home filesystem — paths, partitions
and container images are ilifu's.

## Results in brief

- **Noise-free training target (headline).** Training against the pre-noise signal
  (amplitude AND phase) while conditioning on the noisy observation makes in-hole
  amplitude recoverable: amp MAE 0.208 vs 0.334 for mean-fill at 1.0x SEFD, where a
  noisy-target model ties mean-fill (0.340). The pipeline generates the paired
  clean/noisy data natively (`CLEAN_DATA` snapshot before noise corruption).
- **Delay spectrum.** The fine-tuned model beats the DPSS classical fill in delay space
  with a bootstrap CI clear of zero (all-tiles wlogP-RMSE delta CI [+0.003, +0.077]),
  and the result survives a GPR baseline.
- **Continuum imaging.** Regime-dependent: the model beats classical fills, and the
  flag-fraction sweep maps where inpainting beats flagging outright (holds near ~37%
  flag fraction on realistic faint fields; flagging wins on bright skies).
- **Noise generalisation.** A model trained at 1.0x SEFD evaluates cleanly at 2-4x
  without retraining.

Numbers above are from the pre-restructure verification runs; the `final` experiment
regenerates all of them from scratch with fixed seeds.

## Repo layout

```
run_pipeline.sh            master orchestrator: one command submits the whole experiment
configs/
  telescope/               instrument definitions (meerkat_lband.yaml validated; hera.yaml untested stub)
  experiment/              run parameters: final.yaml (production), massoud_r0-r3, ablation_noisy_target
orchestrator/              config resolver (works without PyYAML on the login node), DAG submit, status
data_preparation/
  simulated/               simms -> crystalball -> CASA noise (+CLEAN_DATA snapshot) -> extract -> inject RFI
  real/                    extract_ms from the flagged MeerKAT MS (tricolour path available)
model/                     conditional DDPM: config.py, unet.py, diffusion.py, data.py, train.py, train_real.py
inference/                 inpaint_infer (GPU preds) + inpaint_write (MS write-back), job queues
evaluation/                delay-spectrum + continuum eval, DPSS/GPR/mean fills, wsclean imaging
figures/                   one script per report figure (see figures/README.md for the map)
tests/                     validation gates: oracle level-0 write-back, hole_pred_check, repr_diag, config_check
docs/                      research notes and audits, indexed with status in docs/README.md
archive/                   superseded eras, each mapped to its replacement in archive/README.md
```

## Quickstart (on ilifu)

```bash
ssh <user>@slurm.ilifu.ac.za
git clone https://github.com/JFMalan/rfi-inpainting-research-pipeline.git
cd rfi-inpainting-research-pipeline
mkdir -p logs

bash run_pipeline.sh configs/experiment/final.yaml --dry-run   # print the 28-stage plan
bash run_pipeline.sh configs/experiment/final.yaml             # submit it
python3 orchestrator/status.py configs/experiment/final.yaml   # stage / jobid / state
```

The orchestrator chains every stage with `sbatch --dependency=afterok` and records
stage -> jobid in `logs/pipeline_<name>.json`. Rerunning after a failure skips completed
stages, reuses queued/running jobs and resubmits failed ones (`--force stage` overrides;
`--only stage1,stage2` submits a subset). Every stage stays individually runnable:

```bash
eval $(python3 orchestrator/resolve_config.py configs/experiment/final.yaml --stage simulate --run 3)
sbatch --export=ALL data_preparation/simulated/jobs/simulate.sh
```

## Regenerating datasets and figures

Datasets live on `/scratch3` (90-day auto-delete) and are regenerable by construction:
every simulate stage is seeded from the experiment config (`seed_base + run`), so
`run_pipeline.sh` reproduces the exact 10 training runs + held-out test run. The real
dataset re-extracts from the source MS on `/idia`. Checkpoints and eval outputs land on
`/idia/users/$USER/rfi/runs/<experiment>_*` (permanent storage).

Figures: `figures/README.md` maps each report figure to its generator script and inputs.
All plot scripts read checkpoints/metrics/npz from disk — no hand-edited plots.

## Pipeline in depth

```
================================================================================================
  CONFIG LAYER (everything below is driven from these two files)
================================================================================================

 configs/telescope/meerkat_lband.yaml          configs/experiment/final.yaml
 ┌──────────────────────────────────┐          ┌─────────────────────────────────────┐
 │ instrument truth:                │          │ run parameters:                     │
 │  simms model, dump 8s, pols      │          │  seed 42, 10+1 runs, synthesis 2.4h │
 │  band 880+856 MHz, extract       │          │  noise draw U(0.7,1.0), RFI draws   │
 │    900-1650 MHz                  │          │  epochs/batch/lr per phase          │
 │  SEFD profile (12 nodes, Jy)     │          │  paths, containers, SLURM resources │
 │  15 persistent-RFI bands         │          │  noise_floor: continuum=none,       │
 │  sky flux scale 0.1-5 Jy         │          │               delay=0.5             │
 └────────────────┬─────────────────┘          └──────────────────┬──────────────────┘
                  └────────────────────┬──────────────────────────┘
                                       v
                 orchestrator/resolve_config.py  (PyYAML, or fallback parser on login node)
                    - merges both files, validates (tests/config_check.py gates the parsers)
                    - deterministic per-run draws: seed=100+N, NOISE_SCALE, TARGET_FRAC,
                      PERSIST_FRAC  (test run pinned at NOISE_SCALE=1.0)
                    - emits per-stage env blocks (RUN_ID=3 SEED=103 SEFD_NODES=... etc)
                                       v
                 orchestrator/submit.py  (via run_pipeline.sh <experiment.yaml>)
                    - builds the 29-stage DAG, submits sbatch --dependency=afterok chains
                    - state file logs/pipeline_final.json; resume skips COMPLETED stages

================================================================================================
  STAGE 1: SIMULATED DATA  —  simulate_run{1..10} + simulate_runtest   (x11, parallel)
  [Main, 128GB, 8h]  data_preparation/simulated/jobs/simulate.sh
================================================================================================

 [0/6] make_random_sky.py (ASTRO-PY3)             seed=100+N
         20-30 point sources, 0.1-5 Jy, spectral index U(-0.9,-0.5), scattered
         around J2000 04h00m -30d  ->  sky_random_N.txt
                     v
 [1/6] simms (stimela_simms container)
         -T meerkat -dt 8 -st 2.4 -f0 880MHz -df 0.8359MHz -nc 1024 -pl "XX YY"
         ->  empty MS: 2016 cross baselines x ~1080 dumps x 1024 chans
                     v
 [2/6] crystalball (codex-africanus container)
         predicts the sky model  ->  DATA column  (clean complex visibilities)
                     v
 [3/6] CASA add_noise.py (casa-stable-v6)                        <- THE CLEAN/NOISY FORK
         DATA ────copy───────────────>  CLEAN_DATA   (pre-noise snapshot = truth)
         sm.corrupt(flat sigma_mean) ->  DATA + CORRECTED_DATA
         + per-channel residual so sigma(nu) = NOISE_SCALE * SEFD(nu)/sqrt(2*dnu*dt)
           (same realization added to both noisy columns; 0.107-0.153 Jy at 1.0x)
                     v
 [4/6] extract_patches_sim.py (ASTRO-PY3)      reads CORRECTED_DATA + CLEAN_DATA
         per baseline (540 x 898 native):
           amp   = mean|V| over pols     phase = angle(mean V)        [noisy]
           divisor = 64-bin running mean of NOISY amp  (divisive norm)
           clean  = noisy_amp / divisor                                [input field]
           amp_target   = clean_amp / SAME noisy divisor               [target field]
           phase_target = clean phase                                  [target field]
         ->  clean_baselines.h5  (2016 baselines, + write-back metadata)
                     v
 [5/6] inject_rfi.py (rfi_toolbox venv)         np.random.seed(SEED)
         synthetic RFI on the native band: narrow/broad persistent+bursty+sweeps,
         persistent-band overlays (drawn PERSIST_FRAC), extra stripes until
         flag frac ~= drawn TARGET_FRAC (0.15-0.50);  corrupted = clean + rfi
         TILING:  2 freq tiles (starts 0,386; 126-ch feathered overlap)
               x  2 non-overlapping 512-bin time windows
         ->  dataset.h5: 8064 units x 512x512
             {clean, corrupted, mask, phase, amp_target, phase_target, dn_divisor,
              baseline_id, ant1, ant2, time_lo, freq_lo, freq_min/max_patch, ...}
                     v
 [6/6] visualise_simulate.py -> vis/: validation checks + sample rows
         [observed (noise, no RFI) | target (no noise, no RFI) | model input (+RFI)]
         with per-patch noise sigma and sky brightness (Jy) on the y axes

================================================================================================
  STAGE 2: REAL DATA  (parallel with sims)
================================================================================================

 extract_real [Main, 128GB, 8h]                      copy_real_ms [Main, 8GB, 8h]
   J2018 corr MS (/idia, read-only,                    cp -r source MS -> /scratch3
   flags already present; tricolour                    (writable copy: write-back adds
   path kept for unflagged sources)                     INPAINTED_DATA/GPR_DATA columns)
   extract_ms.py: same amp/phase/divisor
   recipe, units <=85% flagged (the real
   data is heavily flagged), contiguous
   time runs, persistent bands forced into mask
   ->  /scratch3/.../real/dataset.h5  (no clean truth on real data)

================================================================================================
  STAGE 3: TRAINING
================================================================================================

 The model (shared by both phases):
   x0 (target, 3ch)  = [amp_target, cos(phase_target), sin(phase_target)]     <- CLEAN (phase 1)
   conditioning      = [noisy amp/cos/sin with holes hidden (mean-filled),
                        mask (1ch), freq positional encoding (4ch)]           = 8ch
   x_in              = keep*x0 + mask*x_t          (Palette contract: hole is the only
   loss              = L1 on the HOLE only          thing being denoised; no leak)
   UNet base=64, mult (1,2,4,8,8), attn @ 64/32, T=1000, x0-prediction, EMA

 train_phase1 [GPU, 144h]  <- afterok: simulate_run1..10
   train.py --clean-target on run[0-9]*/dataset.h5 (~72k train units)
   80 epochs, val every 2: complex MAE / amp MAE / PSNR / phase err / noise-floor-ratio
   vs the MEAN-FILL baseline (beats_mf), val sampling at 50 DDIM steps
   ->  /idia/.../runs/final_phase1/best.pt

 train_phase2_finetune [GPU, 144h]  <- afterok: extract_real + train_phase1
 train_phase2_scratch  [GPU, 144h]  <- afterok: extract_real
   train_real.py, self-supervised mixed masking: fake 2D-blob/stripe holes over
   UNFLAGGED pixels (real flags hidden from conditioning, carry no loss);
   noisy target (no clean truth on real data — sim prior carries amplitude);
   40k sample cap, baseline-grouped splits; finetune seeds from final_phase1
   ->  final_phase2_finetune/best.pt  +  final_phase2_scratch/best.pt

================================================================================================
  STAGE 4: INFERENCE + MS WRITE-BACK   (split GPU/CPU on purpose)
================================================================================================

 inpaint_infer.py [GPU, 48GB]:  200-step DDIM, eta=0, conditioning as in training
   noise_floor per arena: none (continuum: grain only pollutes the coherent sum)
                          0.5  (delay: hole texture must match surroundings)
   ->  preds .npz (per-unit amp/cos/sin predictions)

 inpaint_write.py [Main, 128GB]:  inverse of extraction, HOLES ONLY
   amp' = pred_amp * dn_divisor;  V' = amp' * exp(i*atan2(sin,cos))
   feathered blend across the 126-ch tile overlap; RESET_COL guards stale fills;
   row map: row = (time_lo+t)*n_baseline + baseline_id  (verified by oracle gate)
   ->  new MS column (INPAINTED_DATA / INPAINTED_SEL), DATA never overwritten

================================================================================================
  STAGE 5: EVALUATION  —  flagging is the baseline to beat, everywhere
================================================================================================

                          ARENA A: CONTINUUM IMAGE                ARENA B: DELAY SPECTRUM
                          (wsclean, oxkat container)              (per-baseline FFT along nu)
 variants:
   1 flagged             image DATA, holes flagged               fakehole_delay_eval.py:
   2 mean-fill           MEANFILL_DATA column                      fake holes over GOOD data
   3 DPSS fill           DPSSFILL_DATA column                      (known truth), fill with
   4 GPR fill (const-mu) GPR_DATA column                           model / DPSS / GPR / zero,
   5 inpainted (all)     INPAINTED_DATA column                     compare delay spectra
   6 inpainted (select)  INPAINTED_SEL, persistent bands           metrics: wlogP-RMSE,
                         left flagged (KEEP_PERSIST)               hi-delay ratio,
 metrics: off-src RMS, dynamic range,                              1000-draw bootstrap CI
   RMSE vs clean truth (sim) with                                  run on BOTH phase-2 models
   block-bootstrap CI95 (64px tiles)

 image_eval_sim        <- writeback_sim        (test run MS; clean truth exists)
 image_eval_real_all   <- writeback_real_all   (J2018 scratch copy)
 image_eval_real_sel   <- writeback_real_sel
 eval_delay_{finetune,scratch}                 (real dataset, fake holes)
 evaluate_sim          (PSNR/MSE/complex-MAE on held-out runtest -> metrics.json)
 panels_real_{full,selective}  (sim vs finetune vs scratch fills, side by side)

================================================================================================
  VALIDATION GATES (tests/, runnable standalone)         FIGURES (figures/, one per report fig)
================================================================================================
  config_check    parser cross-check                      validate_samples (obs|target|input)
  repr_diag       h5 representation round-trip == 0       train_curves (loss+val vs epoch, mf line)
  oracle_level0   native passthrough == clean in MS       massoud_ladder (rung bars)
  hole_pred_check unit/write-back consistency             plot_width_sweep (crossover)
  smoke/overfit   training sanity                         image/delay comparisons, panels, ...

================================================================================================
  THE DAG AS SUBMITTED (29 stages; -> is afterok, except deps ON TRAINING stages which
  are afterany: a walltime kill still leaves best.pt, so eval continues on the best-so-far
  model; a checkpoint-less crash fails fast on the dependents' guards)
================================================================================================

 simulate_run1 ... simulate_run10 ──────────────> train_phase1 ─────────┬──> evaluate_sim
 simulate_runtest ───────────────────────────────────┬──────────────────┤       (needs runtest too)
                                                     │                  ├──> infer_sim
 extract_real ──┬──> train_phase2_finetune <─────────┼── (also phase1)  │       v
                │         │                          │                  │    writeback_sim
                └──> train_phase2_scratch            │                  │       v
 copy_real_ms ──┐         │                          │                  │    image_eval_sim
                │         ├──> eval_delay_finetune   │
                │         ├──> eval_delay_scratch    │
                │         ├──> panels_real_full + panels_real_selective  (need all 3 ckpts)
                │         └──> infer_real
                │                 ├──> writeback_real_all ──────> image_eval_real_all
                └─────────────────┴──> writeback_real_selective > image_eval_real_selective

 afterwards, from the same datasets/checkpoints (configs/experiment/README.md):
   massoud_r0..r3 (train_phase1 + evaluate_sim each) -> figures/massoud_ladder.py
   ablation_noisy_target (phase1 noisy-target + sim imaging chain)
   R4 / weighted-imaging / native-vs-downsample / width sweep / noise generalisation
```

## Containers (per stage)

| Stage | Container |
|-------|-----------|
| training / GPU inference | `ASTRO-GPU-PyTorch-2026-01-28.sif` (**not** `ASTRO-GPU.simg` — that one has no PyTorch) |
| extraction, write-back, eval python | `ASTRO-PY3.10.sif` |
| thermal noise (CASA sm tool) | `casa-stable-v6.sif` |
| empty MS creation | `STIMELA_IMAGES/stimela_simms_1.2.0.sif` |
| sky prediction | `STIMELA_IMAGES/stimela_codex-africanus_1.6.7.sif` |
| tricolour flagging, wsclean imaging | oxkat container (auto-discovered) |

## Model inventory

| Name | What it is | Path |
|------|------------|------|
| `final_phase1` | sim-trained, noise-free target, 10 runs | `/idia/users/$USER/rfi/runs/final_phase1/best.pt` |
| `final_phase2_finetune` | real MeerKAT, seeded from phase 1 | `/idia/users/$USER/rfi/runs/final_phase2_finetune/best.pt` |
| `final_phase2_scratch` | real MeerKAT, random init (sim-prior contrast) | `/idia/users/$USER/rfi/runs/final_phase2_scratch/best.pt` |
| `massoud_r0..r3_phase1` | component-ladder rungs, fixed run1-3 subset | `/idia/users/$USER/rfi/runs/massoud_r<k>_phase1/best.pt` |
| `phase1_all_old` | pre-restructure sim model (historical) | `/idia/users/$USER/rfi/runs/phase1_all_old/best.pt` |

## Validation gates

`tests/` holds the correctness gates, each runnable standalone: `oracle_level0`
(native-passthrough write-back == clean at the MS level), `hole_pred_check`
(prediction/write-back unit consistency), `repr_diag` (h5 representation round-trip),
`config_check` (login-node config parser vs PyYAML), plus training smoke/overfit checks.

## License and citation

MIT (see LICENSE).

```
@misc{malan2026rfiinpainting,
  author = {Malan, Jacques},
  title  = {RFI inpainting for MeerKAT visibilities with conditional diffusion},
  year   = {2026},
  url    = {https://github.com/JFMalan/rfi-inpainting-research-pipeline}
}
```
