#!/bin/bash
#SBATCH --job-name='rfi-speckle-probe'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/speckle-probe-%j-stdout.log
#SBATCH --error=logs/speckle-probe-%j-stderr.log

set -e

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
DIAG=$SCRIPTS/model/diagnostics
SIM=/scratch3/users/$USER/rfi/simulated
WORK=/scratch3/users/$USER/rfi/realify_test
mkdir -p $WORK logs

LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

CLEAN=${CLEAN:-$SIM/run1/clean_baselines.h5}
[ -f "$CLEAN" ] || CLEAN=$SIM/run1/dataset.h5
echo "source clean: $CLEAN"

ITERS=${ITERS:-3000}
N=${N:-64}
REAL=/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5

echo "=================================================================="
echo "build speckle-sweep variants (each stores clean_smooth = recoverable target)"
echo "=================================================================="
# smooth-component std fixed at 0.10 (real); total = sqrt(0.10^2 + speckle^2)
for SP in 0.00 0.06 0.12 0.18; do
    OUT=$WORK/sweep_sp${SP}.h5
    if [ -f $OUT ]; then echo "$OUT exists, skipping"; continue; fi
    echo "--- speckle_std=$SP ---"
    singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/realify.py \
        --input $CLEAN --output $OUT \
        --amp-std 0.10 --speckle-std $SP --corr-len 1.0 \
        --target-frac 0.48 --band-fill 0.9
done

echo "=================================================================="
echo "visualise speckle vs real FIRST (so images survive even if probes time out)"
echo "=================================================================="
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/visualisation/vis_speckle.py \
    --sim $WORK/sweep_sp0.18.h5 --real $REAL --output $WORK/vis --n 8

echo "=================================================================="
echo "probe: train to plateau, score vs NOISY and SMOOTH target"
echo "=================================================================="
probe () {
    local data=$1 pred=$2 eta=$3
    echo ""
    echo ">>>>> PROBE  $(basename $data)  predict=$pred eta=$eta  iters=$ITERS n=$N"
    singularity exec --nv $NVBIND $GPU python $DIAG/speckle_probe.py \
        --data $data --n $N --iters $ITERS --bs 8 --predict $pred --eta $eta
}

# x0/eta0 is the recovery-optimal config (conditional mean). sweep speckle level.
for SP in 0.00 0.06 0.12 0.18; do
    probe $WORK/sweep_sp${SP}.h5 x0 0.0
done
# at full real speckle, also check eps to confirm it only adds texture, not recovery
probe $WORK/sweep_sp0.18.h5 noise 1.0

echo "done. plots in $WORK/vis/"
