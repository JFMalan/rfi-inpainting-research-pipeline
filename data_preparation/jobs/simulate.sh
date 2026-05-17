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

SIMMS=/idia/software/containers/STIMELA_IMAGES/stimela_simms_1.2.0.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

OUTDIR=/scratch3/users/$USER/rfi/simulated
MS=$OUTDIR/meerkat_sim.ms
WATERFALL=$OUTDIR/waterfall
CLEAN_H5=$OUTDIR/clean_patches.h5
DATASET=$OUTDIR/dataset.h5

mkdir -p $OUTDIR logs

# --- Step 1: create empty MeerKAT L-band MS ---
# 256 channels x 3.125 MHz = 800 MHz bandwidth (880-1680 MHz)
# 1.5h synthesis at 8s dumps = 675 time steps
singularity exec $SIMMS simms \
    -T meerkat \
    -dir "J2000,04h00m00.0s,-30d00m00s" \
    -dt 8 \
    -st 1.5 \
    -f0 880MHz \
    -df 3125kHz \
    -nc 256 \
    -pl "XX YY" \
    -n $MS

# --- Step 2: predict visibilities from sky model using crystalball ---
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate
    crystalball $MS -sm $SCRIPTS/data_preparation/sky_model.txt -o DATA -j 8
"

# --- Step 3a: extract amplitude waterfall from MS ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/extract_ms.py \
    --ms $MS \
    --output $WATERFALL

# --- Step 3b: slice waterfall into 256x256 patches and save HDF5 ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/waterfall_to_patches.py \
    --waterfall ${WATERFALL}.npy \
    --output $CLEAN_H5 \
    --patch-time 256 \
    --patch-freq 256 \
    --stride-time 64 \
    --stride-freq 64 \
    --max-patches 500

# --- Step 4: inject RFI using rfi_toolbox ---
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate
    python $SCRIPTS/data_preparation/inject_rfi.py --input $CLEAN_H5 --output $DATASET --seed 42
"

# --- Step 5: validate ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/validate_simulate.py \
    --input $DATASET
