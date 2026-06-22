#!/bin/bash
#SBATCH --job-name='rfi-regen-real-viz'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/regen-real-viz-%j-stdout.log
#SBATCH --error=logs/regen-real-viz-%j-stderr.log

set -e

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model/diagnostics
DATA=${DATA:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
RUNS=/idia/users/$USER/rfi/runs
DEC_CKPT=${DEC_CKPT:-$RUNS/phase2_decompose/v1_upsample512_finetune/best.pt}
FULL_CKPT=${FULL_CKPT:-$RUNS/phase2_decompose_fullamp/v1_upsample512_finetune/best.pt}
OUT=/idia/users/$USER/rfi/viz
N=${N:-6}
mkdir -p $OUT logs

LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

# decompose ckpt: --smooth-target feeds smooth_component as context (training parity)
echo "==== REAL inpaint, decompose ckpt (smooth-target context) ===="
singularity exec --nv $NVBIND $GPU python $SCRIPTS/inpaint_viz.py \
    --data $DATA --ckpt $DEC_CKPT --out $OUT/real_decompose.npz --real --smooth-target --n $N

echo "==== REAL inpaint, full-amp ckpt (raw context) ===="
singularity exec --nv $NVBIND $GPU python $SCRIPTS/inpaint_viz.py \
    --data $DATA --ckpt $FULL_CKPT --out $OUT/real_fullamp.npz --real --n $N

echo "==== render (trusted-pixel colour scale) ===="
singularity exec $ASTROPY python $SCRIPTS/visualise_samples.py --input "$OUT/real_decompose.npz" --output $OUT --n-show $N
singularity exec $ASTROPY python $SCRIPTS/visualise_samples.py --input "$OUT/real_fullamp.npz" --output $OUT --n-show $N

echo "done -> $OUT/real_decompose.png  $OUT/real_fullamp.png"
