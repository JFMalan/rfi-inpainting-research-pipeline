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
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

# Two Phase-2 configs (override via sbatch --export):
#   (a) fine-tune from sim:  INIT_FROM=/idia/users/$USER/rfi/runs/phase1_run2/best.pt MODE=finetune
#   (b) from scratch:        MODE=scratch  (INIT_FROM unset)
MODE=${MODE:-finetune}
INIT_FROM=${INIT_FROM:-/idia/users/$USER/rfi/runs/phase1_run2/best.pt}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-8}
MAX_PATCHES=${MAX_PATCHES:-}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
DATASET=/scratch3/users/$USER/rfi/real/dataset.h5
OUT=/idia/users/$USER/rfi/runs/phase2_${MODE}

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
if [ -n "$MAX_PATCHES" ]; then EXTRA="--max-patches $MAX_PATCHES"; fi
if [ "$MODE" = "finetune" ]; then EXTRA="$EXTRA --init-from $INIT_FROM"; fi

echo "MODE=$MODE  INIT_FROM=$([ "$MODE" = finetune ] && echo $INIT_FROM || echo none)  EPOCHS=$EPOCHS  BATCH=$BATCH"

singularity exec --nv $NVBIND $GPU python $SCRIPTS/train_real.py \
    --data $DATASET \
    --out $OUT \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    $EXTRA

echo "done"
