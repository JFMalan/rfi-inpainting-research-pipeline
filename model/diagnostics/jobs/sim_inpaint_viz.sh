#!/bin/bash
#SBATCH --job-name='rfi-sim-inpaint-viz'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/sim-inpaint-viz-%j-stdout.log
#SBATCH --error=logs/sim-inpaint-viz-%j-stderr.log

set -e

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
DATA=${DATA:-/scratch3/users/$USER/rfi/simulated/run1/dataset.h5}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_decompose/best.pt}
NPZ=/scratch3/users/$USER/rfi/vis-sim/sim_inpaint.npz
PNG=/scratch3/users/$USER/rfi/vis-sim/sim_inpaint.png

mkdir -p $(dirname $NPZ) logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

[ -f "$CKPT" ] || { echo "checkpoint missing: $CKPT"; exit 1; }

echo "=== inpaint sim with smooth-target model (GT shown = smooth component) ==="
singularity exec --nv $NVBIND $GPU python $SCRIPTS/diagnostics/inpaint_viz.py \
    --data $DATA --ckpt $CKPT --out $NPZ --n ${N:-6} --steps ${STEPS:-200} \
    --smooth-target --smooth-sigma ${SIGMA:-1.0}

echo "=== render ==="
singularity exec $ASTROPY python $SCRIPTS/diagnostics/visualise_samples.py \
    --input $NPZ --output $PNG --n-show ${N:-6}

echo "done -> $PNG"
