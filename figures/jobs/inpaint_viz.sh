#!/bin/bash
#SBATCH --job-name='rfi-inpaint-viz'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A40|V100|A100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/inpaint-viz-%j-stdout.log
#SBATCH --error=logs/inpaint-viz-%j-stderr.log

set -e

CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all/best.pt}
SIM_H5=${SIM_H5:-/scratch3/users/$USER/rfi/simulated/run1/dataset.h5}
REAL_H5=${REAL_H5:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
N=${N:-6}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
FIGS=/users/$USER/rfi-inpainting-research-pipeline/figures
MODEL_SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model/diagnostics
OUT=/idia/users/$USER/rfi/viz
mkdir -p $OUT logs

LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "==== SIM inpaint ===="
singularity exec --nv $NVBIND $GPU python $FIGS/inpaint_viz.py \
    --data $SIM_H5 --ckpt $CKPT --out $OUT/sim_inpaint.npz --n $N

echo "==== REAL inpaint of ACTUAL RFI flags (same sim model) ===="
singularity exec --nv $NVBIND $GPU python $MODEL_SCRIPTS/inpaint_real.py \
    --data $REAL_H5 --ckpt $CKPT --out $OUT/real_inpaint.npz --n $N

echo "==== render both ===="
singularity exec $ASTROPY python $FIGS/visualise_samples.py --input "$OUT/sim_inpaint.npz" --output $OUT --n-show $N
singularity exec $ASTROPY python $FIGS/visualise_real_inpaint.py --input "$OUT/real_inpaint.npz" --output $OUT/real_inpaint.png --n-show $N

echo "done -> $OUT/sim_inpaint.png  $OUT/real_inpaint.png"
