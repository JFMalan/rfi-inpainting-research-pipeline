#!/bin/bash
#SBATCH --job-name='rfi-overfit'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/overfit-%j-stdout.log
#SBATCH --error=logs/overfit-%j-stderr.log

set -e

RUN_ID=${RUN_ID:-1}
N=${N:-256}
ITERS=${ITERS:-8000}
BS=${BS:-12}
LR=${LR:-1e-4}
PREDICT=${PREDICT:-noise}
AMP_ONLY=${AMP_ONLY:-0}
HOLE_FILL=${HOLE_FILL:-mean}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
DATASET=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5

mkdir -p logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

EXTRA=""
if [ "$AMP_ONLY" = "1" ]; then EXTRA="--amp-only"; fi

singularity exec --nv $NVBIND $GPU python $SCRIPTS/overfit_test.py \
    --data $DATASET --n $N --iters $ITERS --bs $BS --lr $LR \
    --predict $PREDICT --hole-fill $HOLE_FILL $EXTRA
