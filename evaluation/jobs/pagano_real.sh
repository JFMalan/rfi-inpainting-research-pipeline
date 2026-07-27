#!/bin/bash
#SBATCH --job-name='rfi-pagano-real'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/pagano-real-%j-stdout.log
#SBATCH --error=logs/pagano-real-%j-stderr.log

set -e

# Pagano et al. 2023 (MNRAS 520, 5552) section 8 real-data method: fake holes are the REAL RFI
# flag mask shifted in frequency onto known-good data, scored with their metrics (amplitude
# fractional error, phase error, and inside/outside-wedge power-spectrum error).
H5=${H5:?set H5=/scratch3/users/$USER/rfi/real/v6_native512.h5}
CKPT=${CKPT:?set CKPT=/idia/users/$USER/rfi/runs/final_phase2_finetune/best.pt}
OUT=${OUT:-/idia/users/$USER/rfi/viz/pagano_real.npz}
SHIFT=${SHIFT:-40}
STEPS=${STEPS:-50}
MAX_UNITS=${MAX_UNITS:-400}
MAX_FLAG=${MAX_FLAG:-0.85}
NOISE_FLOORS=${NOISE_FLOORS:-none 0.5}
WIDE_THRESH=${WIDE_THRESH:-5}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
mkdir -p logs $(dirname $OUT)

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then echo "no driver libs on $(hostname)"; exit 1; fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

singularity exec --nv $NVBIND $GPU python $ROOT/evaluation/pagano_real_eval.py \
    --h5 "$H5" --ckpt "$CKPT" --out "$OUT" \
    --shift $SHIFT --steps $STEPS --max-units $MAX_UNITS --max-flag-frac $MAX_FLAG \
    --noise-floors $NOISE_FLOORS --wide-thresh $WIDE_THRESH

echo "done -> $OUT"
