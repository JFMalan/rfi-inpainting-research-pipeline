#!/bin/bash
#SBATCH --job-name='rfi-realify-test'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/realify-test-%j-stdout.log
#SBATCH --error=logs/realify-test-%j-stderr.log

set -e

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
DIAG=$SCRIPTS/model/diagnostics
SIM=/scratch3/users/$USER/rfi/simulated
REAL=/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5
WORK=/scratch3/users/$USER/rfi/realify_test
mkdir -p $WORK logs

LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

CLEAN=${CLEAN:-$SIM/run1/clean_baselines.h5}
[ -f "$CLEAN" ] || CLEAN=$SIM/run1/dataset.h5
echo "source clean: $CLEAN"

echo "=================================================================="
echo "STEP 1  characterise real speckle (calibration targets)"
echo "=================================================================="
singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/characterise_speckle.py \
    --files $REAL --n 200

echo "=================================================================="
echo "STEP 2  build realify variants"
echo "=================================================================="
SPECKLE=${SPECKLE_STD:-0.18}
CORR=${CORR_LEN:-1.0}

echo "--- variant A: scale+mask only, NO speckle ---"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/realify.py \
    --input $CLEAN --output $WORK/v_nospeckle.h5 \
    --amp-std 0.21 --speckle-std 0.0 --target-frac 0.48 --band-fill 0.9

echo "--- variant B: scale+mask + white speckle (real-calibrated) ---"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/realify.py \
    --input $CLEAN --output $WORK/v_speckle.h5 \
    --amp-std 0.10 --speckle-std $SPECKLE --corr-len $CORR \
    --target-frac 0.48 --band-fill 0.9

echo "=================================================================="
echo "STEP 3  convergence test: x0 vs eps on each variant"
echo "=================================================================="
run_test () {
    local data=$1 pred=$2 eta=$3 tag=$4
    echo ""
    echo ">>>>> TEST [$tag]  data=$(basename $data)  predict=$pred eta=$eta"
    singularity exec --nv $NVBIND $GPU python $DIAG/overfit_test.py \
        --data $data --n 8 --iters 1500 --bs 4 \
        --predict $pred --eta $eta --hole-fill mean
}

run_test $WORK/v_nospeckle.h5 x0    0.0 "nospeckle-x0"
run_test $WORK/v_speckle.h5   x0    0.0 "speckle-x0"
run_test $WORK/v_speckle.h5   noise 1.0 "speckle-eps-eta1"
run_test $WORK/v_speckle.h5   noise 0.0 "speckle-eps-eta0"

echo "done. variants in $WORK/"
