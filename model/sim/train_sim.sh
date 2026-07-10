#!/bin/bash
#SBATCH --job-name='rfi-train'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=144:00:00
#SBATCH --output=logs/train-%j-stdout.log
#SBATCH --error=logs/train-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

RUN_ID=${RUN_ID:-all}
EPOCHS=${EPOCHS:-80}     # tiling doubled the data; 80ep ~117h (fits 144h walltime)
BATCH=${BATCH:-4}
MAX_PATCHES=${MAX_PATCHES:-}
PHASE=${PHASE:-1}
TAG=${TAG:-}            # e.g. _tiled80ep -> OUT phase1_all_tiled80ep (don't overwrite the old phase1_all)
LR=${LR:-}
SEED=${SEED:-}
VAL_EVAL_STEPS=${VAL_EVAL_STEPS:-}
VAL_EVAL_PATCHES=${VAL_EVAL_PATCHES:-}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
# RUN_ID=all -> train on run[0-9]*/dataset.h5 (the diverse multi-run set: run1..run10+,
# excludes runtest and other non-numbered dirs); else one run
if [ "$RUN_ID" = "all" ]; then
    DATASET="/scratch3/users/$USER/rfi/simulated/run[0-9]*/dataset.h5"
else
    DATASET="/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5"
fi
OUT=/idia/users/$USER/rfi/runs/phase${PHASE}_${RUN_ID}${TAG}

mkdir -p $OUT logs

# ilifu singularity.conf has no ldconfig path, so --nv autodetect finds no driver
# libs. Bind the versioned driver libs in manually (filenames vary per node/driver).
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"
echo "node $(hostname)  binding $LIBCUDA  $LIBNVML"

echo "verifying GPU + torch inside container"
singularity exec --nv $NVBIND $GPU python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in container'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

EXTRA=""
if [ -n "$MAX_PATCHES" ]; then EXTRA="--max-patches $MAX_PATCHES"; fi
if [ -n "$LR" ]; then EXTRA="$EXTRA --lr $LR"; fi
if [ -n "$SEED" ]; then EXTRA="$EXTRA --seed $SEED"; fi
if [ -n "$VAL_EVAL_STEPS" ]; then EXTRA="$EXTRA --val-eval-steps $VAL_EVAL_STEPS"; fi
if [ -n "$VAL_EVAL_PATCHES" ]; then EXTRA="$EXTRA --val-eval-patches $VAL_EVAL_PATCHES"; fi
if [ "${CLEAN_TARGET:-0}" = "1" ]; then EXTRA="$EXTRA --clean-target"; fi

singularity exec --nv $NVBIND $GPU python $SCRIPTS/train.py \
    --data "$DATASET" \
    --out $OUT \
    --phase $PHASE \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    $EXTRA

echo "done"
