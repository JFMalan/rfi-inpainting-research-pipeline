#!/bin/bash
#SBATCH --job-name='rfi-stochastic-inpaint'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/stochastic-inpaint-%j-stdout.log
#SBATCH --error=logs/stochastic-inpaint-%j-stderr.log

set -e

# Checkpoint to evaluate. Switch TAG to compare different training runs:
#   phase1_all              = original (noisy target)
#   phase1_all_decompose    = smooth-target training (predict recoverable signal)
TAG=${TAG:-phase1_all_decompose}
N=${N:-64}
STEPS=${STEPS:-200}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/${TAG}/best.pt}
DATA=${DATA:-/scratch3/users/$USER/rfi/simulated/runtest/dataset.h5}
OUT=/scratch3/users/$USER/rfi/diagnostics/stochastic_inpaint/${TAG}

if [ ! -f "$CKPT" ]; then echo "checkpoint not found: $CKPT"; exit 1; fi

mkdir -p logs "$OUT"
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"
echo "node $(hostname)  tag=$TAG  ckpt=$CKPT  data=$DATA"

singularity exec --nv $NVBIND $GPU python $ROOT/model/diagnostics/stochastic_inpaint.py \
    --ckpt    "$CKPT" \
    --data    "$DATA" \
    --n       $N \
    --steps   $STEPS \
    --predict x0 \
    --out-png "$OUT/grid.png"

echo "done -> $OUT/"
