#!/bin/bash
#SBATCH --job-name='rfi-sweep'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/sweep-%j-stdout.log
#SBATCH --error=logs/sweep-%j-stderr.log

set -e

RUN_ID=${RUN_ID:-1}
N=${N:-256}
ITERS=${ITERS:-3000}
BS=${BS:-12}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model/diagnostics
DATASET=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5

mkdir -p logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

run () {
    echo ""
    echo "=================================================================="
    echo "CONFIG: $1"
    echo "=================================================================="
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/overfit_test.py \
        --data $DATASET --n $N --iters $ITERS --bs $BS $2 || echo "  (config failed)"
}

echo "### RECOVERABILITY (no training) — is masked amplitude predictable at all?"
singularity exec $GPU python $SCRIPTS/recoverability.py --data $DATASET --n 200 || true

echo ""
echo "### MODEL SWEEP — n=$N iters=$ITERS bs=$BS"
run "baseline: joint 3ch, predict=noise, hole=zero, lr=1e-4"        "--lr 1e-4 --predict noise --hole-fill zero"
run "hole-fill=mean (fix the OOD cliff)"                            "--lr 1e-4 --predict noise --hole-fill mean"
run "hole-fill=center"                                              "--lr 1e-4 --predict noise --hole-fill center"
run "predict=x0 + hole=mean"                                        "--lr 1e-4 --predict x0    --hole-fill mean"
run "amp-only + hole=mean"                                          "--lr 1e-4 --predict noise --hole-fill mean --amp-only"
run "amp-only + predict=x0 + hole=mean"                             "--lr 1e-4 --predict x0    --hole-fill mean --amp-only"
run "lower lr 5e-5 + hole=mean"                                     "--lr 5e-5 --predict noise --hole-fill mean"

echo ""
echo "### DONE — compare 'closed X% of gap' and MODEL MAE vs mean-fill across configs"
