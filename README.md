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
