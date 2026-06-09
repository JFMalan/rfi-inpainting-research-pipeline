#!/bin/bash
#SBATCH --job-name='rfi-train'
#SBATCH --partition=GPU
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/train-%j-stdout.log
#SBATCH --error=logs/train-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

RUN_ID=${RUN_ID:-1}
EPOCHS=${EPOCHS:-400}
BATCH=${BATCH:-16}
MAX_PATCHES=${MAX_PATCHES:-}
PHASE=${PHASE:-1}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
DATASET=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5
OUT=/idia/users/$USER/rfi/runs/phase${PHASE}_run${RUN_ID}

mkdir -p $OUT logs

echo "verifying GPU + torch inside container"
singularity exec --nv $GPU python -c "import torch; assert torch.cuda.is_available(), 'no CUDA in container'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

EXTRA=""
if [ -n "$MAX_PATCHES" ]; then EXTRA="--max-patches $MAX_PATCHES"; fi

singularity exec --nv $GPU python $SCRIPTS/train.py \
    --data $DATASET \
    --out $OUT \
    --phase $PHASE \
    --epochs $EPOCHS \
    --batch-size $BATCH \
    $EXTRA

echo "done"
