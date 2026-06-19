#!/bin/bash
#SBATCH --job-name='rfi-train-decompose'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A40
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=72:00:00
#SBATCH --output=logs/train-decompose-%j-stdout.log
#SBATCH --error=logs/train-decompose-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

RUN_ID=${RUN_ID:-all}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-4}
MAX_PATCHES=${MAX_PATCHES:-}
PHASE=${PHASE:-1}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
if [ "$RUN_ID" = "all" ]; then
    DATASET="/scratch3/users/$USER/rfi/simulated/run[1-9]/dataset.h5"
else
    DATASET="/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5"
fi
OUT=/idia/users/$USER/rfi/runs/phase${PHASE}_${RUN_ID}_decompose

mkdir -p $OUT logs

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"
echo "node $(hostname)  binding $LIBCUDA  $LIBNVML"

singularity exec --nv $NVBIND $GPU python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))"

EXTRA=""
if [ -n "$MAX_PATCHES" ]; then EXTRA="--max-patches $MAX_PATCHES"; fi

# --smooth-target: predict the recoverable smooth bandpass (decompose-then-inpaint).
# val metrics (amp_mae, complex_mae) are then measured against the smooth target.
singularity exec --nv $NVBIND $GPU python $SCRIPTS/train.py \
    --data "$DATASET" \
    --out $OUT \
    --phase $PHASE \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    --smooth-target \
    $EXTRA

echo "done"
