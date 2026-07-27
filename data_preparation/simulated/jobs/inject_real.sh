#!/bin/bash
#SBATCH --job-name='rfi-inject-real'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/inject-real-%j-stdout.log
#SBATCH --error=logs/inject-real-%j-stderr.log

set -e

# Realistic (stochastic) RFI injection over clean sim baselines: bursty + intermittent + persistent
# bands + frequency sweeps, topped up to TARGET_FRAC. Unlike inject_width.sh this activates the
# rfi_toolbox venv, which the SyntheticDataGenerator needs (controlled/full-time-stripe mode does
# not, which is why inject_width.sh omits it). Output drops straight into inpaint_infer / image_eval.
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
ROOT=/users/$USER/rfi-inpainting-research-pipeline
CLEAN_H5=${CLEAN_H5:?set CLEAN_H5=/scratch3/users/$USER/rfi/simulated/<run>/clean_baselines.h5}
OUT=${OUT:?set OUT=/path/to/dataset.h5}
TARGET_FRAC=${TARGET_FRAC:-0.37}
SEED=${SEED:-42}

mkdir -p logs $(dirname $OUT)
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate &&
    python $ROOT/data_preparation/simulated/inject_rfi.py \
        --input '$CLEAN_H5' --output '$OUT' --target-frac $TARGET_FRAC --seed $SEED
"

echo "done -> $OUT"
