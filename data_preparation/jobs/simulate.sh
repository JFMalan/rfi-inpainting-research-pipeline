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

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
CASA=/idia/software/containers/casa-stable-v6.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

SIMDIR=/scratch3/users/$USER/rfi/simulated
SIM_MS=$SIMDIR/sim_clean.ms
WATERFALL=$SIMDIR/waterfall
CLEAN_H5=$SIMDIR/clean_patches.h5
DATASET=$SIMDIR/dataset.h5

mkdir -p $SIMDIR logs

# --- Step 1: generate synthetic clean MeerKAT MS via Stimela2 ---
singularity exec $ASTROPY stimela run \
    $SCRIPTS/data_preparation/simulate_vis.yml simulate-meerkat \
    ms-name=$SIM_MS \
    sky-model=$SCRIPTS/data_preparation/sky_model.txt \
    n-hours=0.5

# --- Step 2: add MeerKAT thermal noise to synthetic MS (SEFD~430 Jy, sigma~0.105 Jy/vis) ---
singularity exec $CASA casa --nologger --log2term \
    -c $SCRIPTS/data_preparation/add_noise.py $SIM_MS

# --- Step 3: extract amplitude waterfall from synthetic MS ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/extract_ms.py \
    --ms $SIM_MS \
    --output $WATERFALL

# --- Step 4: slice waterfall into 256x256 patches ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/waterfall_to_patches.py \
    --waterfall ${WATERFALL}.npy \
    --output $CLEAN_H5 \
    --patch-time 256 \
    --patch-freq 256 \
    --stride-time 64 \
    --stride-freq 64 \
    --max-patches 500 \
    --max-flag-fraction 0.5

# --- Step 5: inject synthetic RFI using rfi_toolbox ---
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate
    python $SCRIPTS/data_preparation/inject_rfi.py --input $CLEAN_H5 --output $DATASET --seed 42
"

# --- Step 6: validate ---
singularity exec $ASTROPY python $SCRIPTS/data_preparation/validate_simulate.py \
    --input $DATASET
