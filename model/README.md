# Model — Conditional DDPM (Palette) for RFI inpainting

Phase 1 trains supervised on simulated paired data (`dataset.h5` from `data_preparation/simulated`).
Phase 2 (mixed masking on real data) reuses the same core; hooks are stubbed in `config.py` / `metrics.py`.

## Files
- `data.py` — `PatchDataset` over one or more `dataset.h5` files, frequency positional encoding, augmentation, conditioning stack
- `unet.py` — conditional U-Net (timestep embedding, residual blocks, self-attention at 32/16, skip connections)
- `diffusion.py` — cosine-schedule DDPM, masked-L1 objective, RePaint-style masked sampling
- `config.py` — `phase1()` / `phase2()` factories
- `metrics.py` — MAE, PSNR (mask region); `tre()` is a phase-2 stub
- `train.py` — training loop, EMA, checkpointing, periodic sampling
- `jobs/train_sim.sh` — SLURM GPU job

## Input contract (from inject_rfi.py)
`dataset.h5` holds `clean`, `corrupted`, `mask` each `(N, 256, 256)` float32 (time × freq), plus
per-patch `freq_min_patch` / `freq_max_patch` and attrs `n_time`, `n_freq`, `freq_min_mhz`, `freq_max_mhz`.

Network input channels = `noisy x_t (1) + corrupted (1) + mask (1) + PE (pe_channels)`.
The positional encoding uses each patch's absolute frequency bounds, so a patch at 1200 MHz and one at
1500 MHz get different PE — this is what breaks translation symmetry along frequency.

## Before launching anything — inspect the data
The data-prep memory is weeks old. Confirm a dataset exists and is well-formed first:

```bash
ls -lh /scratch3/users/$USER/rfi/simulated/run1/dataset.h5
singularity exec /idia/software/containers/ASTRO-PY3.10.sif python - <<'PY'
import h5py, numpy as np
f = h5py.File('/scratch3/users/$USER/rfi/simulated/run1/dataset.h5','r')
for k in ['clean','corrupted','mask']:
    print(k, f[k].shape, f[k].dtype)
m = f['mask'][:200]
print('patches:', f['clean'].shape[0])
print('mask frac mean:', m.mean())
print('clean amp mean/std:', f['clean'][:200].mean(), f['clean'][:200].std())
print('has freq_min_patch:', 'freq_min_patch' in f)
PY
```
Expect ~hundreds of patches, mask fraction ~0.05–0.40, clean amplitude O(0.1–1) (divisive-normalised).
If patch count is only a few hundred this is fine for the smoke test below — DDPMs want thousands,
so generate more runs (`simulate.sh` with different `SEED`/`SKY_MODEL`) before a real training run and
point `--data` at a glob like `'.../run*/dataset.h5'`.

## Smoke test (one run, prove the loop learns)
```bash
cd /users/$USER/rfi-inpainting-research-pipeline
sbatch --export=ALL,RUN_ID=1,EPOCHS=300,BATCH=8 model/jobs/train_sim.sh
```
The job uses `ASTRO-GPU-PyTorch-2026-01-28.sif` (the bare `ASTRO-GPU.simg` is TensorFlow/JAX — no torch).
It asserts CUDA + torch inside the GPU allocation and prints the device, then trains. The CUDA check
only passes on a GPU node — running it on a login/compute node reports False because there is no GPU there.
Watch the loss in `logs/train-<jobid>-stdout.log`; MAE/PSNR (mask region) are logged every
`sample_every` epochs and sample `.npz` files written to the output dir for visual inspection.

Outputs go to `/idia/users/$USER/rfi/runs/phase1_run1/` (`ckpt.pt`, `log.json`, `samples/`).

## Inspect reconstructions
The training loop writes `samples/sample_e<epoch>.npz` (clean / corrupted / mask / pred). Render them:
```bash
singularity exec /idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif \
    python model/visualise_samples.py \
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
