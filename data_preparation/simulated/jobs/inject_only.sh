#!/bin/bash
#SBATCH --job-name='rfi-inject'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/inject-%j-stdout.log
#SBATCH --error=logs/inject-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

RUN_ID=${RUN_ID:-1}
SEED=${SEED:-42}

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

SIMDIR=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}
WATERFALL=$SIMDIR/waterfall
CLEAN_H5=$SIMDIR/clean_patches.h5
DATASET=$SIMDIR/dataset.h5
VISDIR=$SIMDIR/vis

mkdir -p $VISDIR logs

if [ ! -f "$CLEAN_H5" ]; then
    echo "missing $CLEAN_H5 — run the full simulate.sh first"; exit 1
fi

echo "[1/2] $(date '+%H:%M:%S') injecting synthetic RFI into $CLEAN_H5"
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate &&
    python $SCRIPTS/data_preparation/simulated/inject_rfi.py \
        --input $CLEAN_H5 --output $DATASET --seed $SEED
"

echo "[2/2] $(date '+%H:%M:%S') validating + visualising"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/visualisation/visualise_simulate.py \
    --input $DATASET \
    --output $VISDIR \
    --waterfall ${WATERFALL}.npy \
    --n-plot 6 \
    --n-patches-show 50

echo "done $(date '+%H:%M:%S')"
echo "dataset -> $DATASET"
