#!/bin/bash
#SBATCH --job-name='rfi-simulate'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/simulate-%j-stdout.log
#SBATCH --error=logs/simulate-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

SIMMS=/idia/software/containers/STIMELA_IMAGES/stimela_simms_1.2.0.sif
AFRICANUS=/idia/software/containers/STIMELA_IMAGES/stimela_codex-africanus_1.6.7.sif
CASA=/idia/software/containers/casa-stable-v6.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

SIMDIR=/scratch3/users/$USER/rfi/simulated
SIM_MS=$SIMDIR/sim_clean.ms
WATERFALL=$SIMDIR/waterfall
CLEAN_H5=$SIMDIR/clean_patches.h5
DATASET=$SIMDIR/dataset.h5

mkdir -p $SIMDIR logs

# --- Step 1: create empty MeerKAT MS ---
singularity exec $SIMMS simms \
    -T meerkat \
    -dir "J2000,04h00m00.0s,-30d00m00s" \
    -dt 8 \
    -st 0.25 \
    -f0 880MHz \
    -df 835.9375kHz \
    -nc 1024 \
    -pl "XX YY" \
    -n $SIM_MS

# --- Step 2: predict sky model into DATA column ---
singularity exec $AFRICANUS crystalball \
    -sm $SCRIPTS/data_preparation/sky_model.txt \
    -o DATA \
    -rc 10000 \
    -mc 25 \
    -j 8 \
    $SIM_MS

# --- Step 3: add MeerKAT thermal noise (SEFD~430 Jy, sigma~0.105 Jy/vis) ---
singularity exec $CASA casa --nologger --log2term \
    -c $SCRIPTS/data_preparation/add_noise.py $SIM_MS

# --- Step 4: extract amplitude waterfall ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/extract_ms.py \
    --ms $SIM_MS \
    --output $WATERFALL

# --- Step 5: slice into 256x256 patches ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/waterfall_to_patches.py \
    --waterfall ${WATERFALL}.npy \
    --output $CLEAN_H5 \
    --patch-time 256 \
    --patch-freq 256 \
    --stride-time 64 \
    --stride-freq 64 \
    --max-patches 100 \
    --max-flag-fraction 0.5

# --- Step 6: inject synthetic RFI ---
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate &&
    python $SCRIPTS/data_preparation/inject_rfi.py \
        --input $CLEAN_H5 --output $DATASET --seed 42
"

# --- Step 7: validate ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/validate_simulate.py \
    --input $DATASET
