#!/bin/bash
#SBATCH --job-name='rfi-finetune-v1'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/finetune-v1-%j-stdout.log
#SBATCH --error=logs/finetune-v1-%j-stderr.log

set -e

# Decisive Phase-2 test: does sim-pretraining beat mean-fill on real where
# from-scratch could not (v1 from-scratch tied mean-fill at TRE 5.40)?
INIT=${INIT:-/idia/users/$USER/rfi/runs/phase1_all/best.pt}
ITERS=${ITERS:-3000}
BATCH=${BATCH:-4}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
H5=/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5
OUT=/idia/users/$USER/rfi/runs/phase2_finetune_v1

mkdir -p $OUT logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

if [ ! -f "$INIT" ]; then
    echo "sim checkpoint not found: $INIT (is the sim training done?)"; exit 1
fi

echo "======== FINE-TUNE v1 from sim $INIT ========"
singularity exec --nv $NVBIND $GPU python $SCRIPTS/train_real.py \
    --data $H5 --out $OUT --init-from $INIT \
    --epochs 1000 --batch-size $BATCH --max-iters $ITERS \
    --sample-every 4 --val-eval-patches 24 --min-epochs 4 --min-delta 0.02 --patience 4

echo "======== EVAL (same held-out test as the from-scratch v1) ========"
singularity exec --nv $NVBIND $GPU python $SCRIPTS/real/eval_real.py \
    --data $H5 --ckpt $OUT/best.pt --tag v1_finetune --batch-size $BATCH

echo ""
echo "compare to from-scratch v1: TRE 5.3966 / mean-fill 5.3721 / fakeMAE 0.1949 / mf 0.1620"
