#!/bin/bash
#SBATCH --job-name='rfi-train-real'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/train-real-%j-stdout.log
#SBATCH --error=logs/train-real-%j-stderr.log

set -e

# Phase-2 job, config-driven: MODE=finetune seeds from the sim checkpoint, MODE=scratch
# trains from random init. Both are always run for the report (sim-prior contrast).
MODE=${MODE:-finetune}
DATA=${DATA:-/scratch3/users/$USER/rfi/real_patches.h5}
OUT=${OUT:-/idia/users/$USER/rfi/runs/phase2_${MODE}}
INIT_FROM=${INIT_FROM:-/idia/users/$USER/rfi/runs/final_phase1/best.pt}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-8}
LR=${LR:-}
SEED=${SEED:-}
MAX_PATCHES=${MAX_PATCHES:-}
EMA_DECAY=${EMA_DECAY:-}
VAL_EVAL_PATCHES=${VAL_EVAL_PATCHES:-}
VAL_EVAL_STEPS=${VAL_EVAL_STEPS:-}
FAKE_MASK_MODE=${FAKE_MASK_MODE:-}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
mkdir -p $OUT logs

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"
echo "node $(hostname)  binding $LIBCUDA  $LIBNVML"

singularity exec --nv $NVBIND $GPU python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in container'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

EXTRA=""
# auto-resume after a walltime kill; RESUME=0 forces a fresh run
if [ "${RESUME:-1}" = "1" ] && [ -f "$OUT/ckpt.pt" ]; then
    echo "resuming from $OUT/ckpt.pt"
    RESUMING=1
    EXTRA="--resume $OUT/ckpt.pt"
fi
if [ -z "$RESUMING" ] && [ -f "$OUT/best.pt" ]; then
    if [ "${FRESH_OK:-0}" = "1" ]; then
        mv "$OUT/best.pt" "$OUT/best_prev.pt"
        echo "fresh start: existing best.pt backed up to best_prev.pt"
    else
        echo "REFUSING fresh start: $OUT/best.pt exists and no ckpt.pt to resume from."
        echo "A fresh run would overwrite it at the first eval. Set FRESH_OK=1 to back it"
        echo "up to best_prev.pt and proceed, or point OUT at a new directory."
        exit 1
    fi
fi
if [ "$MODE" = "finetune" ]; then
    if [ ! -f "$INIT_FROM" ]; then echo "sim checkpoint not found: $INIT_FROM"; exit 1; fi
    EXTRA="$EXTRA --init-from $INIT_FROM"
fi
if [ -n "$LR" ]; then EXTRA="$EXTRA --lr $LR"; fi
if [ -n "$SEED" ]; then EXTRA="$EXTRA --seed $SEED"; fi
if [ -n "$MAX_PATCHES" ]; then EXTRA="$EXTRA --max-patches $MAX_PATCHES"; fi
if [ -n "$EMA_DECAY" ]; then EXTRA="$EXTRA --ema-decay $EMA_DECAY"; fi
if [ -n "$VAL_EVAL_PATCHES" ]; then EXTRA="$EXTRA --val-eval-patches $VAL_EVAL_PATCHES"; fi
if [ -n "$VAL_EVAL_STEPS" ]; then EXTRA="$EXTRA --val-eval-steps $VAL_EVAL_STEPS"; fi
if [ -n "$FAKE_MASK_MODE" ]; then EXTRA="$EXTRA --fake-mask-mode $FAKE_MASK_MODE"; fi

echo "MODE=$MODE  DATA=$DATA  OUT=$OUT  EPOCHS=$EPOCHS  BATCH=$BATCH  EXTRA=$EXTRA"

singularity exec --nv $NVBIND $GPU python $SCRIPTS/train_real.py \
    --data $DATA \
    --out $OUT \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    $EXTRA

echo "done"
