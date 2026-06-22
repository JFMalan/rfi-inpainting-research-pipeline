#!/bin/bash
#SBATCH --job-name='rfi-compare-inpaint'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/compare-inpaint-%j-stdout.log
#SBATCH --error=logs/compare-inpaint-%j-stderr.log

set -e

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
DATA=${DATA:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
SIM=${SIM:-/idia/users/$USER/rfi/runs/phase1_all_decompose/best.pt}
FT=${FT:-/idia/users/$USER/rfi/runs/phase2_decompose/v1_upsample512_finetune/best.pt}
SCRATCH=${SCRATCH:-/idia/users/$USER/rfi/runs/phase2_decompose/v1_upsample512_scratch/best.pt}
OUT=${OUT:-/scratch3/users/$USER/rfi/vis-compare/sim_vs_finetune.png}

mkdir -p $(dirname $OUT) logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

for C in "$SIM" "$FT"; do
    [ -f "$C" ] || { echo "checkpoint missing: $C"; exit 1; }
done
# scratch is optional — include the column only if its checkpoint exists
SCRATCH_ARG=""
if [ -f "$SCRATCH" ]; then SCRATCH_ARG="--scratch-ckpt $SCRATCH"; else echo "no scratch ckpt at $SCRATCH, skipping that column"; fi
# RESAMPLE=1 (default) adds speckle for smooth-target models so fills blend; set 0 for
# full-amplitude models (they predict their own texture — show it raw).
RESAMPLE_ARG=""
if [ "${RESAMPLE:-1}" = "1" ]; then RESAMPLE_ARG="--resample-speckle --smooth-sigma ${SIGMA:-1.0}"; fi

singularity exec --nv $NVBIND $GPU python $SCRIPTS/diagnostics/compare_inpaint.py \
    --data $DATA --sim-ckpt $SIM --ft-ckpt $FT $SCRATCH_ARG --output $OUT --n ${N:-5} --steps ${STEPS:-200} \
    $RESAMPLE_ARG

echo "done -> $OUT"
