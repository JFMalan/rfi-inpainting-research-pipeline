#!/bin/bash
#SBATCH --job-name='rfi-r4-simdelay'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/r4-simdelay-%j-stdout.log
#SBATCH --error=logs/r4-simdelay-%j-stderr.log

set -e

# Ladder rung R4 = R3 + sampling/write-back techniques (inference-only). On simulated
# runtest with noise-free clean truth, sweep the sampling noise_floor and measure
# delay-space recovery vs DPSS/GPR/flagged.
H5=${H5:?set H5=/scratch3/users/$USER/rfi/simulated/runtest/dataset.h5}
CKPT=${CKPT:?set CKPT=/idia/users/$USER/rfi/runs/massoud_r3_phase1/best.pt}
OUT=${OUT:-/idia/users/$USER/rfi/runs/massoud_r4_eval/sim_delay.npz}
STEPS=${STEPS:-50}
MAX_UNITS=${MAX_UNITS:-300}
NOISE_FLOORS=${NOISE_FLOORS:-none 0.3 0.5 auto}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
mkdir -p logs $(dirname $OUT)

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then echo "no driver libs on $(hostname)"; exit 1; fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

singularity exec --nv $NVBIND $GPU python $ROOT/evaluation/sim_delay_eval.py \
    --h5 "$H5" --ckpt "$CKPT" --out "$OUT" \
    --steps $STEPS --max-units $MAX_UNITS --noise-floors $NOISE_FLOORS

echo "done -> $OUT"
