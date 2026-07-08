#!/bin/bash
#SBATCH --job-name='rfi-fill-check'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/fill-check-%j-stdout.log
#SBATCH --error=logs/fill-check-%j-stderr.log

set -e
GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
H5=${H5:?set H5 (paired dataset.h5)}
CKPT=${CKPT:?set CKPT (best.pt trained on the paired dataset)}
OUT=${OUT:-/idia/users/$USER/rfi/viz/noise_threshold/fill_check.png}
STEPS=${STEPS:-50}
N=${N:-4}
mkdir -p logs $(dirname $OUT)

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then echo "no driver libs on $(hostname)"; exit 1; fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

singularity exec --nv $NVBIND $GPU python $ROOT/model/diagnostics/noise_free_fill_check.py \
    --h5 "$H5" --ckpt "$CKPT" --output "$OUT" --steps $STEPS --n-show $N
echo "done -> $OUT"
