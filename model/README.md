# Model — Conditional DDPM (Palette) for RFI inpainting

Phase 1 trains supervised on simulated paired data (`dataset.h5` from `data_preparation/simulated`).
Phase 2 (mixed masking on real data) reuses the same core; hooks are stubbed in `config.py` / `metrics.py`.

The model is **3-channel**: it reconstructs amplitude *and* phase so the inpainted patch can be turned
back into a complex visibility and written into an MS. The target/output is `(3, T, F)`:
- **ch0 — amplitude**, divisively normalised to unit scale (mean≈1).
- **ch1 — cos(phase)**, **ch2 — sin(phase)**. Phase is carried as a cos/sin pair, not the raw angle, to
  avoid the ±π wrap discontinuity. Recover the angle with `atan2(ch2, ch1)`.

RFI is injected into amplitude only, so the two phase channels are identical in `clean` and `corrupted`;
the model still has to denoise/inpaint all three under the diffusion objective.

## Layout
Shared core (used by both phases) lives at the `model/` root; phase-specific entry points and one-off
diagnostics are in subfolders.

**Core**
- `data.py` — `PatchDataset` over one or more `dataset.h5` files, frequency positional encoding, augmentation, 3-channel target stacking, conditioning stack
- `unet.py` — conditional U-Net (timestep embedding, residual blocks, self-attention at 32/16, skip connections)
- `diffusion.py` — cosine-schedule DDPM, masked-L1 objective, RePaint-style masked sampling (optional `progress` callback)
- `config.py` — `phase1()` / `phase2()` factories
- `metrics.py` — amplitude MAE/PSNR (mask region, ch0) + `phase_error`; `tre()` is a phase-2 stub
- `train.py` — training loop, EMA, checkpointing, periodic sampling

**`sim/`** — simulation-trained (Phase 1) entry point
- `sim/train_sim.sh` — SLURM GPU job for supervised training on simulated `dataset.h5`

**`real/`** — raw-data (Phase 2) entry point (mixed masking on real MeerKAT data; stubs)

**`diagnostics/`** — one-off investigation/test scripts built during Phase 1 (not part of the training
pipeline). Each adds `sys.path` to import the core. Notable: `pipeline_doctor.py` (stage isolation),
`info_ceiling.py` / `recoverability.py` (is structure recoverable), `stochastic_inpaint.py`
(eta/noise-floor + texture ratio), `speckle_probe.py`, `overfit_test.py`, `visualise_samples.py`
(rendering). Their SLURM wrappers are in `diagnostics/jobs/`. Superseded scripts (`bias_diag.py`,
`infer_compare.py`, `gen_sweep.py`, `sampler_sweep.py`, `viz_eta.py`) live in `archive/`.

## Input contract
`dataset.h5` is produced by `data_preparation/simulated` — `extract_patches_sim.py` writes the clean
patches and all metadata, `inject_rfi.py` adds `corrupted` and `mask`. The model **requires** all of the
following (see [data_preparation/simulated/README.md](../data_preparation/simulated/README.md) for the
full schema):

| Dataset | Shape | Why the model / write-back needs it |
|---------|-------|-------------------------------------|
| `clean` | `(N, 256, 256)` f32 | normalised amplitude ground truth (ch0 target) |
| `corrupted` | `(N, 256, 256)` f32 | RFI-injected amplitude (conditioning) |
| `mask` | `(N, 256, 256)` f32 | RFI mask, 1 = corrupted pixel |
| `phase` | `(N, 256, 256)` f32 | per-pixel phase; split into cos/sin to form ch1/ch2 and to rebuild the complex visibility |
| `dn_divisor` | `(N, 256, 256)` f32 | the divisive-norm smoothing curve; multiply ch0 by it to invert the normalisation back to physical Jy |
| `freq_min_patch` / `freq_max_patch` | `(N,)` f32 | absolute frequency bounds of each patch → positional encoding |
| `chan_offset` / `time_offset` | `(N,)` i32 | channel/time origin of the patch in the full waterfall → where to write the inpainted patch back |
| `baseline_id` / `ant1` / `ant2` | `(N,)` i32 | which baseline (and its antennas) the patch came from → MS row placement |

Attrs: `freq_min_mhz`, `freq_max_mhz`, `n_time`, `n_freq`, `n_patches`, `full_n_time`, `full_n_chan`,
`chan_lo`, `n_baseline`.

`phase`, `dn_divisor`, and the offset/baseline fields are only consumed by the model loader for the
phase channels (`phase`); the rest exist so the eventual MS write-back can de-normalise and re-place each
patch. They are carried through `inject_rfi.py` untouched.

Network input channels = **11** = `noisy x_t (3) + masked conditioning (3) + mask (1) + PE (4)`, output
channels = 3 (`config.in_channels` / `target_channels`). The positional encoding uses each patch's
absolute frequency bounds, so a patch at 1200 MHz and one at 1500 MHz get different PE — this is what
breaks translation symmetry along frequency.

**True inpainting, not RFI-removal.** `build_cond` zeroes the masked (RFI) pixels of the conditioning
(`known = corrupted * (1 - mask)`) so the network never sees the corruption itself — it must fill the
holes from the surrounding clean context. The mask channel marks where the holes are. An earlier setup
fed the raw `corrupted` patch as conditioning, which let the model learn to subtract a visible
corruption (RFI-removal) and leaked the answer; that is fixed.

## Inspect the data (verify before launching)
Confirm the dataset before a long run (note: `$USER` does not expand inside a `<<'PY'` heredoc — read
the path from the environment in Python):
```bash
ls -lh /scratch3/users/$USER/rfi/simulated/run1/dataset.h5
singularity exec /idia/software/containers/ASTRO-PY3.10.sif python - <<'PY'
import os, h5py
f = h5py.File(f"/scratch3/users/{os.environ['USER']}/rfi/simulated/run1/dataset.h5", 'r')
for k in ['clean', 'corrupted', 'mask', 'phase', 'dn_divisor']:
    print(k, f[k].shape, f[k].dtype)
print('patches:', f['clean'].shape[0])
print('mask frac mean:', float(f['mask'][:200].mean()))
print('clean amp mean/std:', float(f['clean'][:200].mean()), float(f['clean'][:200].std()))
for k in ['phase', 'dn_divisor', 'chan_offset', 'time_offset', 'baseline_id']:
    print('has', k + ':', k in f)
PY
```
A correct run holds ~100,800 patches, mask fraction ~0.12, clean amplitude mean≈1.0 std≈0.31
(divisive normalisation centred it at unit scale — no extra normalisation is applied or needed). This is
a full training set; no additional runs are required for phase 1. To use several runs, point `--data` at
a glob like `'.../run*/dataset.h5'` (the loader concatenates them).

## Regenerate the dataset first (required)
**The old single-channel `dataset.h5` will not work.** It has no `phase` (the loader reads `f['phase']`
unconditionally) and no `dn_divisor`/offset/baseline fields. Re-run `data_preparation/simulated`
end-to-end to produce a 3-channel dataset before training.

The previous **~43 dB amplitude-only PSNR is superseded** and should not be reported: it came from the
single-channel model *and* from the leaky conditioning (the corruption was visible to the network, so it
was learning RFI-removal rather than inpainting). Both are fixed here. Treat all numbers from this point
as the first valid baseline.

## Quick correctness check first (~minutes, optional)
Prove the loop runs end-to-end on a small subset before committing GPU days:
```bash
cd /users/$USER/rfi-inpainting-research-pipeline
sbatch --export=ALL,RUN_ID=1,EPOCHS=4,BATCH=16,MAX_PATCHES=512 model/sim/train_sim.sh
```

## Full phase-1 training
```bash
cd /users/$USER/rfi-inpainting-research-pipeline
sbatch --export=ALL,RUN_ID=1 model/sim/train_sim.sh        # defaults: 40 epochs, batch 32
```
### GPU notes (verified on ilifu)
- **P100 nodes (gpu-001–004) are unusable** — CUDA capability sm_60, torch 2.10 needs sm_70+.
- Usable: V100 (gpu-005, 32 GB), A40 (gpu-006, 48 GB), A100 (gpu-007, 40 GB). Constraint is `A100|A40|V100`.
- **Batch 32 OOMs on the V100's 32 GB.** Use `BATCH=16` on the V100; batch 32 is fine on A40/A100.
- `singularity --nv` finds no driver libs on these nodes (`ldconfig not set in singularity.conf`); the job
  manually binds the versioned `libcuda`/`libnvidia-ml` (auto-detected) — see the top of the job script.
- It uses `ASTRO-GPU-PyTorch-2026-01-28.sif` (the bare `ASTRO-GPU.simg` is TensorFlow/JAX — no torch).

At batch 16, one epoch over 20k patches is ~1,250 batches ≈ 23 min on the V100 (~15 h for 40 epochs).

### Data splits
`PatchDataset` deterministically splits the full pool into train/val/test (90/5/5%, fixed `split_seed`,
disjoint, stable across runs and resumes). `MAX_PATCHES` caps only the **train** split — val and test
always come from the full held-out pools. The MAE/PSNR in the training log are on **val** (one batch,
noisy — for watching the trend only). The reportable result comes from `evaluation/evaluate.py` on the
**test** split. Do not report the in-training val numbers as the final figure.

### Metrics
MAE and PSNR are computed on **amplitude only** (channel 0) inside the mask region — these are the
reportable figures. `metrics.phase_error` reports mean absolute angular error (radians) over the mask,
recovered from the cos/sin channels via `atan2`; report it separately, do not fold it into the
amplitude PSNR. `tre()` stays a phase-2 stub (real data, no clean truth).

Outputs go to `/idia/users/$USER/rfi/runs/phase1_run1/` (`ckpt.pt`, `log.json`, `samples/`).

## Test-set evaluation (the reportable result)
After training, evaluate on the held-out **test** split. Each patch needs a full 1000-step DDPM reverse
pass, so sampling is slow — start with a subset (~512 patches gives a stable mean):
```bash
sbatch --export=ALL,RUN_ID=1,SPLIT=test,MAX_EVAL=512 evaluation/jobs/eval.sh
```
For the full 5,040-patch test set drop `MAX_EVAL` and raise the walltime. Results land in
`runs/phase1_run1/eval_test/metrics.json` (mean ± std MAE/PSNR) with sample `.npz` files.

## Inspect reconstructions
The training loop writes `samples/sample_e<epoch>.npz` (clean / corrupted / mask / pred). Render them:
```bash
singularity exec /idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif \
    python figures/visualise_samples.py \
    --input '/idia/users/$USER/rfi/runs/phase1_run1/samples/*.npz' \
    --output /idia/users/$USER/rfi/runs/phase1_run1/sample_pngs --n-show 6
```
Each row: clean truth | corrupted+mask | prediction | per-pixel error inside the mask (the part that
actually matters). If the predicted panel reproduces the clean structure under the green mask, it works.

## Container
Use `ASTRO-GPU-PyTorch-2026-01-28.sif`, which ships torch + h5py + numpy + scipy. The bare
`ASTRO-GPU.simg` and the `ASTRO-GPU-TF-*` images are TensorFlow/JAX stacks without torch. There are
older PyTorch images (`ASTRO-GPU-PyTorch-2023.09.sif` etc.) if the 2026 one ever misbehaves.

## Phase 2 (later)
`phase2()` enables mixed fake-masking and the masked-loss formulation already wired in `diffusion.loss`
(`loss_region='mask'`, weak global term via `mask_weight`). Implement fake-mask injection in the dataset,
the statistical-alignment sampler tweak in `diffusion.sample`, and `metrics.tre` for real data.
