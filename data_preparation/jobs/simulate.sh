#!/bin/bash
#SBATCH --job-name='rfi-simulate'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/simulate-%j-stdout.log
#SBATCH --error=logs/simulate-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

MS=/idia/data/public/1491550051/1491550051.ms
OUTDIR=/scratch3/users/$USER/rfi/simulated
WATERFALL=$OUTDIR/waterfall
CLEAN_H5=$OUTDIR/clean_patches.h5
DATASET=$OUTDIR/dataset.h5

mkdir -p $OUTDIR logs

# --- Step 1: extract amplitude waterfall from real MeerKAT MS ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/extract_ms.py \
    --ms $MS \
    --output $WATERFALL

# --- Step 2: slice waterfall into 256x256 patches, skip heavily flagged patches ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/waterfall_to_patches.py \
    --waterfall ${WATERFALL}.npy \
    --output $CLEAN_H5 \
    --patch-time 256 \
    --patch-freq 256 \
    --stride-time 64 \
    --stride-freq 64 \
    --max-patches 500 \
    --max-flag-fraction 0.5

# --- Step 3: inject synthetic RFI using rfi_toolbox ---
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate
    python $SCRIPTS/data_preparation/inject_rfi.py --input $CLEAN_H5 --output $DATASET --seed 42
"

# --- Step 4: validate ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/validate_simulate.py \
    --input $DATASET
