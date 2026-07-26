#!/bin/bash
#SBATCH --job-name='rfi-eval-sim'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval-sim-%j-stdout.log
#SBATCH --error=logs/eval-sim-%j-stderr.log

set -e

H5=${H5:?set H5=/scratch3/users/$USER/rfi/simulated/runtest/dataset.h5}
CKPT=${CKPT:?set CKPT=/path/to/best.pt}
OUT=${OUT:?set OUT=/path/to/eval_out_dir}
SPLIT=${SPLIT:-all}
STEPS=${STEPS:-50}
MAX_EVAL=${MAX_EVAL:-}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
mkdir -p logs $OUT

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then echo "no driver libs on $(hostname)"; exit 1; fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

EXTRA=""
[ -n "$MAX_EVAL" ] && EXTRA="--max-eval $MAX_EVAL"

singularity exec --nv $NVBIND $GPU python $ROOT/evaluation/evaluate.py \
    --data "$H5" --ckpt "$CKPT" --out "$OUT" --split $SPLIT --steps $STEPS $EXTRA

echo "done -> $OUT"
