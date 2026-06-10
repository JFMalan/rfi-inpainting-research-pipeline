#!/bin/bash
#SBATCH --job-name='rfi-eval'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval-%j-stdout.log
#SBATCH --error=logs/eval-%j-stderr.log

set -e

RUN_ID=${RUN_ID:-1}
PHASE=${PHASE:-1}
SPLIT=${SPLIT:-test}
BATCH=${BATCH:-16}
MAX_EVAL=${MAX_EVAL:-}
WEIGHTS=${WEIGHTS:-best.pt}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
DATASET=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5
RUNDIR=/idia/users/$USER/rfi/runs/phase${PHASE}_run${RUN_ID}
CKPT=$RUNDIR/$WEIGHTS
if [ ! -f "$CKPT" ]; then CKPT=$RUNDIR/ckpt.pt; fi
OUT=/idia/users/$USER/rfi/runs/phase${PHASE}_run${RUN_ID}/eval_${SPLIT}

mkdir -p $OUT logs

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"
echo "node $(hostname)  evaluating $SPLIT split  ckpt $CKPT"

EXTRA=""
if [ -n "$MAX_EVAL" ]; then EXTRA="--max-eval $MAX_EVAL"; fi

singularity exec --nv $NVBIND $GPU python $ROOT/evaluation/evaluate.py \
    --data $DATASET \
    --ckpt $CKPT \
    --out $OUT \
    --split $SPLIT \
    --batch-size $BATCH \
    $EXTRA

echo "done"
