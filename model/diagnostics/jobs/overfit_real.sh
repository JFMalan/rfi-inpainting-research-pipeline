#!/bin/bash
#SBATCH --job-name='rfi-overfit-real'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/overfit-real-%j-stdout.log
#SBATCH --error=logs/overfit-real-%j-stderr.log

set -e

# SYNTHETIC=1 -> in-memory structured data, no dataset needed (tests the wiring).
# SYNTHETIC=0 -> overfit real baselines from $DATASET (tests it learns real holes).
SYNTHETIC=${SYNTHETIC:-1}
N=${N:-8}
ITERS=${ITERS:-400}
BS=${BS:-4}
LR=${LR:-2e-4}
PREDICT=${PREDICT:-x0}
HOLE_FILL=${HOLE_FILL:-mean}
ETA=${ETA:-0.0}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model/diagnostics
DATASET=/scratch3/users/$USER/rfi/real/dataset.h5

mkdir -p logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

EXTRA=""
if [ "$SYNTHETIC" = "1" ]; then EXTRA="--synthetic"; else EXTRA="--data $DATASET"; fi

singularity exec --nv $NVBIND $GPU python $SCRIPTS/overfit_real.py \
    --n $N --iters $ITERS --bs $BS --lr $LR \
    --predict $PREDICT --hole-fill $HOLE_FILL --eta $ETA $EXTRA
