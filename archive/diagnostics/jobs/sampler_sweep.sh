#!/bin/bash
#SBATCH --job-name='rfi-sweep'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep-%j-stdout.log
#SBATCH --error=logs/sweep-%j-stderr.log

set -e

RUN_ID=${RUN_ID:-2}
PHASE=${PHASE:-1}
N=${N:-8}
PREDICT=${PREDICT:-x0}
STEPS=${STEPS:-1000}
WEIGHTS=${WEIGHTS:-best.pt}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
DATASET=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5
RUNDIR=/idia/users/$USER/rfi/runs/phase${PHASE}_run${RUN_ID}
CKPT=$RUNDIR/$WEIGHTS
if [ ! -f "$CKPT" ]; then CKPT=$RUNDIR/ckpt.pt; fi
OUT=$RUNDIR/sweep.npz

mkdir -p logs
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"
echo "node $(hostname)  sweep ckpt $CKPT"

singularity exec --nv $NVBIND $GPU python $ROOT/archive/diagnostics/sampler_sweep.py \
    --data $DATASET --ckpt $CKPT --out $OUT \
    --n $N --predict $PREDICT --steps $STEPS \
    --etas 0.0 0.5 1.0 --repaint-u 1 5

echo "rendering"
singularity exec $GPU python $ROOT/archive/diagnostics/viz_eta.py --input $OUT --output $RUNDIR/sweep.png --n $N
echo "done -> $RUNDIR/sweep.png"
