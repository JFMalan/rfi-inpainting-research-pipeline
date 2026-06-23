#!/bin/bash
#SBATCH --job-name='stochastic-inpaint'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/stochastic-inpaint-%j.log
#SBATCH --error=logs/stochastic-inpaint-%j.log

set -e

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

# Switch CKPT to compare different training runs:
#   phase1_all        = original (trained on noisy target — no smooth_target)
#   phase1_all_decompose_80ep = smooth-target training (predict recoverable signal)
TAG=${TAG:-phase1_all_decompose_80ep}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/${TAG}/best.pt}
DATA=${DATA:-/scratch3/users/$USER/rfi/simulated/run1/dataset.h5}
OUT=/scratch3/users/$USER/rfi/diagnostics/stochastic_inpaint/${TAG}

mkdir -p logs "$OUT"

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

echo "node $(hostname)  ckpt=$CKPT  data=$DATA"

singularity exec --nv $NVBIND $GPU \
    python $SCRIPTS/model/diagnostics/stochastic_inpaint.py \
        --ckpt    "$CKPT" \
        --data    "$DATA" \
        --n       128 \
        --steps   200 \
        --predict x0 \
        --out-png "$OUT/grid.png"

echo "results in $OUT/"
