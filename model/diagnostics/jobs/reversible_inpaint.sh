#!/bin/bash
#SBATCH --job-name='rfi-reversible-inpaint'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/reversible-inpaint-%j-stdout.log
#SBATCH --error=logs/reversible-inpaint-%j-stderr.log

set -e

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
DATA=${DATA:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
# the smooth-target finetune (the model that learned recoverable structure)
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase2_decompose/v1_upsample512_finetune/best.pt}
OUT=${OUT:-/scratch3/users/$USER/rfi/vis-reversible/reversible_inpaint.png}

mkdir -p $(dirname $OUT) logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT"; exit 1; }

singularity exec --nv $NVBIND $GPU python $SCRIPTS/diagnostics/reversible_inpaint.py \
    --data $DATA --ckpt $CKPT --output $OUT \
    --methods ${METHODS:-gaussian,median,wavelet} --sigma ${SIGMA:-1.0} \
    --n ${N:-4} --steps ${STEPS:-200}

echo "done -> $OUT"
