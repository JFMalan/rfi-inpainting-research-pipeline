# RFI Inpainting Research Pipeline

Conditional DDPM (Palette architecture) for in-painting RFI-corrupted MeerKAT radio spectrograms.

## Overview

Train a conditional diffusion model to reconstruct RFI-flagged visibilities in MeerKAT spectrograms, recovering data that would otherwise be discarded. Two training phases:

1. **Supervised** — simulated clean data with injected RFI (ground truth known)
2. **Self-supervised** — real MeerKAT MS data using mixed masking (no clean ground truth needed)

Key techniques: Palette U-Net backbone, L1 loss, divisive normalisation, sine-based positional encoding, RePaint temporal resampling.

Metrics: MAE and PSNR (simulated), TRE (real data).

## Data

Real data comes from existing MeerKAT Measurement Sets on ilifu (`/idia/data/public/`, `/idia/raw/`). Simulated data is generated via Stimela with RFI injected using the van Zyl framework.

## RFI Flagging

Uses **tricolour** (not AOFlagger) — GPU/dask-accelerated, built for MeerKAT, writes flags back to the MS `FLAG` column in-place.

## Execution

All runs happen on **ilifu** via SLURM inside Singularity containers. GPU training uses `ASTRO-GPU.simg` on the GPU partition.

## Status

Pipeline is being built from scratch on ilifu.
