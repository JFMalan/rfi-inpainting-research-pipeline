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

## Inspect the data (verify before launching)
Confirm the dataset before a long run (note: `$USER` does not expand inside a `<<'PY'` heredoc — read
the path from the environment in Python):
```bash
ls -lh /scratch3/users/$USER/rfi/simulated/run1/dataset.h5
singularity exec /idia/software/containers/ASTRO-PY3.10.sif python - <<'PY'
import os, h5py
f = h5py.File(f"/scratch3/users/{os.environ['USER']}/rfi/simulated/run1/dataset.h5", 'r')
for k in ['clean', 'corrupted', 'mask']:
    print(k, f[k].shape, f[k].dtype)
print('patches:', f['clean'].shape[0])
print('mask frac mean:', float(f['mask'][:200].mean()))
print('clean amp mean/std:', float(f['clean'][:200].mean()), float(f['clean'][:200].std()))
print('has freq_min_patch:', 'freq_min_patch' in f)
PY
```
run1 currently holds **100,800 patches**, mask fraction ~0.12, clean amplitude mean≈1.0 std≈0.31
(divisive normalisation centred it at unit scale — no extra normalisation is applied or needed). This is
a full training set; no additional runs are required for phase 1. To use several runs, point `--data` at
a glob like `'.../run*/dataset.h5'` (the loader concatenates them).

## Quick correctness check first (~minutes, optional)
Prove the loop runs end-to-end on a small subset before committing GPU days:
```bash
cd /users/$USER/rfi-inpainting-research-pipeline
sbatch --export=ALL,RUN_ID=1,EPOCHS=4,BATCH=16,MAX_PATCHES=512 model/jobs/train_sim.sh
```

## Full phase-1 training
```bash
cd /users/$USER/rfi-inpainting-research-pipeline
sbatch --export=ALL,RUN_ID=1 model/jobs/train_sim.sh        # defaults: 40 epochs, batch 32
```
The job requests an A100 or A40 (`--constraint=A100|A40`) — batch 32 at 256² needs ≥32 GB, so the 12 GB
P100 nodes are excluded. Drop `BATCH` to ~8 and remove the constraint if you must run on a P100/V100.
It uses `ASTRO-GPU-PyTorch-2026-01-28.sif` (the bare `ASTRO-GPU.simg` is TensorFlow/JAX — no torch),
asserts CUDA + torch inside the GPU allocation, prints the device, then trains. The CUDA check only
passes on a GPU node. One epoch is ~3,150 batches at batch 32 — expect hours per epoch depending on GPU.
Watch the loss in `logs/train-<jobid>-stdout.log`; mask-region MAE/PSNR are logged every `sample_every`
epochs (default 2) and sample `.npz` files written for visual inspection.

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
