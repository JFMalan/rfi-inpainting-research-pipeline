#!/bin/bash
#SBATCH --job-name='rfi-compare-models'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/compare-models-%j-stdout.log
#SBATCH --error=logs/compare-models-%j-stderr.log

set -e

# Amplitude inpaint comparison on REAL held-out tiles: sim (phase1) vs finetune vs scratch,
# same tiles, same noise_floor. Fills the actual RFI flags so the panels show real inpainting.
GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
RUNS=/idia/users/$USER/rfi/runs
H5=${H5:-/scratch3/users/$USER/rfi/real/variants/v6_native512.h5}
SIM_CKPT=${SIM_CKPT:-$RUNS/phase1_all_tiled80ep/best.pt}
FT_CKPT=${FT_CKPT:-$RUNS/phase2_decompose_fullamp/v6_native512_finetune/best.pt}
SC_CKPT=${SC_CKPT:-$RUNS/phase2_decompose_fullamp/v6_native512_scratch/best.pt}
OUT=${OUT:-/idia/users/$USER/rfi/viz/compare_models_real.png}
NF=${NF:-0.5}
N=${N:-20}
STEPS=${STEPS:-50}

for c in "$SIM_CKPT" "$FT_CKPT" "$SC_CKPT"; do
    [ -f "$c" ] || { echo "checkpoint not found: $c"; exit 1; }
done

mkdir -p logs $(dirname $OUT)
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then echo "no driver libs on $(hostname)"; exit 1; fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

singularity exec --nv $NVBIND $GPU python $ROOT/model/diagnostics/compare_models_real.py \
    --h5 "$H5" --output "$OUT" --noise-floor $NF --n-show $N --steps $STEPS \
    --ckpts sim="$SIM_CKPT" finetune="$FT_CKPT" scratch="$SC_CKPT"

echo "done -> $OUT"
