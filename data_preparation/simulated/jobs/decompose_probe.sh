#!/bin/bash
#SBATCH --job-name='rfi-decompose-probe'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/decompose-probe-%j-stdout.log
#SBATCH --error=logs/decompose-probe-%j-stderr.log

set -e

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
DIAG=$SCRIPTS/model/diagnostics
SIM=/scratch3/users/$USER/rfi/simulated
WORK=/scratch3/users/$USER/rfi/decompose_test
mkdir -p $WORK logs

LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

CLEAN=${CLEAN:-$SIM/run1/clean_baselines.h5}
[ -f "$CLEAN" ] || CLEAN=$SIM/run1/dataset.h5
echo "source clean: $CLEAN"

ITERS=${ITERS:-3000}
N=${N:-64}

# full-std structure (0.21, the recoverable signal at REAL scale) + white speckle on top.
# clean_smooth stores the std-0.21 structure; clean = structure + 0.18 white noise.
OUT=$WORK/full_struct_sp0.18.h5
if [ -f $OUT ]; then echo "$OUT exists, skipping build"; else
echo "=== build full-structure + speckle variant ==="
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/realify.py \
    --input $CLEAN --output $OUT \
    --amp-std 0.21 --speckle-std 0.18 --corr-len 1.0 \
    --target-frac 0.48 --band-fill 0.9
fi

probe () {
    local pred=$1 eta=$2 tt=$3 tag=$4
    echo ""
    echo ">>>>> PROBE [$tag]  predict=$pred eta=$eta train-target=$tt"
    singularity exec --nv $NVBIND $GPU python $DIAG/speckle_probe.py \
        --data $OUT --n $N --iters $ITERS --bs 4 --eval-bs 4 \
        --predict $pred --eta $eta --train-target $tt
}

# control: train on noisy target (reproduces the old failure mode at full structure)
probe x0 0.0 noisy  "noisy-target-control"
# the discovery: train on the recoverable smooth structure, context = noisy obs
probe x0 0.0 smooth "smooth-target-DECOMPOSE"
# decompose + stochastic texture resample on top
probe noise 1.0 smooth "smooth-target-eps-texture"

echo "done."
