# Restructure proposal — target layout and config schemas

Checkpoint 2 of the final restructure. Companion to `restructure-inventory.md`.

## Design principle

Existing top-level module names stay (`data_preparation/`, `model/`, `inference/`,
`evaluation/`) — the production scripts have cluster-tested import paths and job scripts
reference `/users/$USER/rfi-inpainting-research-pipeline/<dir>` on ilifu. Renaming those
directories re-debugs the whole pipeline for zero scientific gain. The restructure adds
the missing areas (`configs/`, `figures/`, `tests/`, `orchestrator/`), empties the dead
weight into `archive/`, and unifies parameters into the two YAML levels.

## Target tree

```
rfi-inpainting-research-pipeline/
├── run_pipeline.sh                  # master orchestrator entry: run_pipeline.sh configs/experiment/final.yaml
├── README.md                        # rewritten: results, layout, quickstart, ilifu assumption
├── configs/
│   ├── telescope/
│   │   ├── meerkat_lband.yaml       # validated instance
│   │   └── hera.yaml                # documented stub, untested (uvh5 ingest noted)
│   └── experiment/
│       ├── final.yaml               # the production experiment: 10+1 sim runs, both phase-2 configs, full eval
│       ├── massoud_r0.yaml          # ablation rungs (R1-R3 as small deltas on r0)
│       └── ...                      # one yaml per ablation that needs a retrain/re-eval
├── orchestrator/
│   ├── resolve_config.py            # YAML -> flat KEY=VALUE env block / CLI args per stage
│   ├── submit.py                    # builds the sbatch afterok chain, writes state file
│   └── status.py                    # state file -> stage/jobid/status table
├── data_preparation/
│   ├── tiling.py
│   ├── simulated/                   # simulate.sh pipeline + extraction + injection (production set only)
│   └── real/                        # flag_real, extract_ms, rfi_bands, tricolour yaml, visualisation
├── model/                           # config.py, data.py, diffusion.py, unet.py, metrics.py, train.py, train_real.py, jobs/
├── inference/                       # inpaint_infer.py, inpaint_write.py, jobs/
├── evaluation/                      # classical_fill, delay_spectrum, fakehole_delay_eval, image_eval, fill writers, jobs/
├── figures/                         # one script per report figure + figures/jobs/; writes into images/
├── images/                          # committed report figures (outputs of figures/, regenerable)
├── tests/                           # validation gates: oracle_level0, hole_pred_check, repr_diag,
│   └── jobs/                        #   pipeline_doctor, smoke_test, overfit_test + their launchers
├── docs/                            # working docs + README.md status index (nothing deleted)
└── archive/                         # superseded scripts, grouped by era, README maps each to its replacement
```

## How YAML reaches the scripts (no rewrite of working code)

The Python scripts keep their CLI flags; the SLURM job scripts keep their env-var
override convention. The orchestrator is the only config consumer:
`resolve_config.py` merges telescope + experiment YAML and emits the env block each
stage is submitted with (`sbatch --export=...`). Stage scripts stay individually
runnable by hand exactly as now — the YAML just becomes the single place the values
come from.

Two contained exceptions where instrument constants currently live inside Python and
must start reading from the telescope YAML (small, isolated edits):

- `add_noise.py` — SEFD nodes and dump time move to `telescope.sefd` / `telescope.dump_time_s`
  (fixes the silent 8s duplication between `simms -dt` and `delta_t`).
- `rfi_bands.py` — becomes a loader for `telescope.persistent_rfi_mhz` (the 15-range
  oxkat list is ground truth; the coarse 2-range summary in the docs is stale).

## configs/telescope/meerkat_lband.yaml (full draft)

```yaml
name: meerkat_lband
ingest: ms                       # HERA stub will say uvh5

simms_model: meerkat             # simms -T
polarizations: [XX, YY]
dump_time_s: 8.0                 # feeds simms -dt AND the add_noise sigma; single source now

band:
  sim_f0_mhz: 880.0              # simulated band start (simms -f0)
  sim_bandwidth_mhz: 856.0       # df = bandwidth / nchan
  extract_min_mhz: 900.0         # analysis band (replaces 5 independent redeclarations)
  extract_max_mhz: 1650.0

sefd:                            # Mauch et al. 2020 + commissioning
  nodes_mhz: [856, 900, 950, 1000, 1100, 1280, 1400, 1450, 1550, 1600, 1650, 1712]
  nodes_jy:  [560, 510, 450, 420, 390, 390, 400, 420, 450, 470, 500, 560]

persistent_rfi_mhz:              # oxkat / MeerKAT Cookbook emitter list (rfi_bands.py verbatim)
  - [900, 915]
  - [925, 960]
  - [1080, 1095]
  - [1166, 1186]
  - [1191, 1217]
  - [1217, 1237]
  - [1242, 1249]
  - [1260, 1300]
  - [1375, 1387]
  - [1453, 1490]
  - [1526, 1554]
  - [1565, 1585]
  - [1592, 1610]
  - [1616, 1626]
  - [1599, 1601]

static_mask_uvrange: "0~550"     # tricolour short-baseline static mask

sky:                             # realistic per-run random skies (make_random_sky.py)
  reference_freq_hz: 1.28e9
  pointing: { ra_h: 4.0, dec_deg: -30.0 }
  flux_min_jy: 0.1
  flux_max_jy: 5.0
  n_src: [20, 30]
  spectral_index: [-0.9, -0.5]
  scatter: { ra_h: 0.15, dec_deg: 1.0 }
```

## configs/experiment/final.yaml (full draft)

```yaml
name: final
telescope: meerkat_lband
seed: 42

paths:
  scratch: /scratch3/users/$USER/rfi
  runs: /idia/users/$USER/rfi/runs
  repo: /users/$USER/rfi-inpainting-research-pipeline

containers:
  pytorch:   /idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif   # NOT ASTRO-GPU.simg (no torch)
  astro_py3: /idia/software/containers/ASTRO-PY3.10.sif
  casa:      /idia/software/containers/casa-stable-v6.sif
  simms:     /idia/software/containers/stimela_simms_1.2.0.sif
  africanus: /idia/software/containers/stimela_codex-africanus_1.6.7.sif
  oxkat: auto                    # discovered by grep at runtime, as now

sim:
  n_train_runs: 10
  test_run: true                 # run 0 = held out, noise pinned 1.0x
  seed_base: 100                 # run r uses seed_base + r
  synthesis_h: 2.4               # >= 2 non-overlapping 512-bin time windows at 8s dumps
  nchan: 1024
  img_size: 512
  noise_scale: [0.7, 1.0]        # uniform draw per training run
  rfi:                           # diversity axis: draw per run
    target_flag_frac: [0.15, 0.50]
    persist_frac: [0.3, 0.8]
    scale: [5.0, 50.0]
  random_sky: true               # sky_model generated per run (main diversity axis)

extract:
  smooth_bins: 64                # divisive-norm running mean (shared sim + real)
  sim_max_bl_flag_frac: 0.8
  clean_target: true             # snapshot pre-noise DATA; amp + phase_target fields, noisy divisor shared

real:
  ms: /idia/projects/astro-cirg/data_for_rfi/1570802018_sdp_l0-J2018_5539-corr.ms
  field: ACT-CLJ2023.3-5535      # tricolour -fn (per-observation metadata)
  column: DATA
  max_bl_flag_frac: 0.5
  max_sample_flag_frac: 0.85     # exclude samples >85% flagged
  max_samples: 40000

train:
  phase1: { epochs: 80, batch: 4, lr: 2.0e-4, val_eval_steps: 50, val_eval_patches: 64 }
  phase2:
    modes: [finetune, scratch]   # always both; sim-prior contrast is a report result
    epochs: 60
    batch: 8
    lr: 4.0e-5
    ema_decay: auto              # run-length scaled; set a number to pin

inference:
  steps: 200
  batch: 8
  noise_floor:                   # per-arena, applied automatically
    continuum: none
    delay: 0.5                   # 2026-07-04 sweep verdict: posterior under-dispersed, 0.5 beats matched;
                                 # ablation still sweeps {none, 0.3, 0.5, matched}

writeback:
  out_col: INPAINTED_DATA
  reset_col: true                # stale-fill landmine
  feather: true

eval:
  variants: [flagged, dpss, gpr, inpaint_all, inpaint_selective]
  imaging: { imsize: 2048, cell: 2asec, niter: 10000, weight: "briggs -0.3",
             auto_mask: 4, auto_threshold: 1, mgain: 0.9 }
  delay:   { dpss_hw: 0.1, dpss_lam: 0.1, gpr_ell: 30.0, gpr_noise: 0.05,
             fg_bins: 20, bootstrap: 1000 }

slurm:                           # per-stage overrides; defaults baked into stage definitions
  simulate:  { partition: Main, mem: 128GB, time: "08:00:00", cpus: 8 }
  extract:   { partition: Main, mem: 128GB, time: "08:00:00", cpus: 8 }
  flag:      { partition: Main, mem: 128GB, time: "10:00:00", cpus: 32 }
  train:     { partition: GPU, gpus: 1, constraint: "A100|A40|V100", mem: 64GB, time: "144:00:00", cpus: 8 }
  infer:     { partition: GPU, gpus: 1, constraint: "A100|A40|V100", mem: 48GB, time: "08:00:00", cpus: 8 }
  writeback: { partition: Main, mem: 128GB, time: "04:00:00", cpus: 8 }
  image:     { partition: Main, mem: 64GB, time: "04:00:00", cpus: 16 }
```

Ablation experiments are separate small YAMLs that override the relevant block
(e.g. `massoud_r0.yaml` sets `train.phase1` flags for the R0 recipe and points at the
fixed 2-3 run subset); rungs R1-R3 are deltas on that file.

## Resolutions to the awkward cases found in the parameter audit

| Case | Resolution |
|---|---|
| dump time hardcoded twice (simms -dt, add_noise delta_t) | single `telescope.dump_time_s` feeds both |
| band edges redeclared 5+ times | `telescope.band.extract_min/max_mhz` only source |
| max_bl_flag_frac 0.8 (sim) vs 0.5 (real) | intentional, kept as two distinct keys |
| persistent bands: code (15 ranges) vs docs (2 ranges) | code list is ground truth in YAML; docs note updated |
| flag_real.sh has zero env overrides | gains MS/FIELD env vars like its siblings |
| tricolour field name is per-observation | lives in `experiment.real.field` |
| wsclean params hardcoded in image_eval.sh | `experiment.eval.imaging` block |
| RFI_SCALE constants + CLI duplicating | YAML value passed via existing CLI flags |
| smooth_bins copy-pasted sim/real | one `extract.smooth_bins` key |
| EMA auto-scale formula | `ema_decay: auto` keeps the formula; a number pins it |
| img_size vs attn_res coupling | resolve_config asserts compatibility at submit time |
| sky pointing/ref-freq duplicated | `telescope.sky` only source |
