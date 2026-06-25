#!/bin/bash
#SBATCH --job-name='rfi-reextract'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/reextract-%j-stdout.log
#SBATCH --error=logs/reextract-%j-stderr.log

set -e

# Re-extract an EXISTING sim run (steps 4-5 of simulate.sh) on its already-simulated sim_clean.ms,
# without re-running simms/crystalball/add_noise. extract now stores the native per-baseline
# waterfall; inject adds RFI on the native band then tiles to img_size (overlapping native-512 freq
# tiles, time cropped to 512), so dataset.h5 has n_tiles x more units. Same SEED reproduces the RFI.
# Loop RUN_ID over the training runs.
RUN_ID=${RUN_ID:-1}
SEED=${SEED:-42}
IMG_SIZE=${IMG_SIZE:-512}
TARGET_FRAC=${TARGET_FRAC:-0.37}

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

SIMDIR=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}
SIM_MS=$SIMDIR/sim_clean.ms
WATERFALL=$SIMDIR/waterfall
CLEAN_H5=$SIMDIR/clean_baselines.h5
DATASET=$SIMDIR/dataset.h5
mkdir -p logs

if [ ! -d "$SIM_MS" ]; then echo "sim MS missing: $SIM_MS (re-run simulate.sh for this run)"; exit 1; fi
echo "RUN_ID=$RUN_ID  ms=$SIM_MS  seed=$SEED"

echo "[4/5] $(date '+%H:%M:%S') extracting native per-baseline waterfalls"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/extract_patches_sim.py \
    --ms $SIM_MS --output $CLEAN_H5 --waterfall-out $WATERFALL \
    --freq-min 900 --freq-max 1650

echo "[5/5] $(date '+%H:%M:%S') injecting synthetic RFI on the native band and tiling to ${IMG_SIZE} (seed $SEED)"
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate &&
    python $SCRIPTS/data_preparation/simulated/inject_rfi.py \
        --input $CLEAN_H5 --output $DATASET --img-size $IMG_SIZE --seed $SEED --target-frac $TARGET_FRAC
"

echo "done $(date '+%H:%M:%S') -> $DATASET"
