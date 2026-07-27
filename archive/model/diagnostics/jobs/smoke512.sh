#!/bin/bash
#SBATCH --job-name='rfi-smoke512'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke512-%j-stdout.log
#SBATCH --error=logs/smoke512-%j-stderr.log

set -e

RUN_ID=${RUN_ID:-1}
PREDICT=${PREDICT:-x0}
OVERFIT=${OVERFIT:-1}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
SCRIPTS=$ROOT/tests
DATASET=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5

mkdir -p logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "=== batch/timing sweep ==="
singularity exec --nv $NVBIND $GPU python $ROOT/archive/model/diagnostics/smoke512.py \
    --data $DATASET --predict $PREDICT

if [ "$OVERFIT" = "1" ]; then
    echo ""
    echo "=== overfit learning test (8 baselines, 400 iters) ==="
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/overfit_test.py \
        --data $DATASET --n 8 --iters 400 --bs 4 --predict $PREDICT --eta 0.0
fi
