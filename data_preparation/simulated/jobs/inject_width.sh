#!/bin/bash
#SBATCH --job-name='rfi-inject-width'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/inject-width-%j-stdout.log
#SBATCH --error=logs/inject-width-%j-stderr.log

set -e

# Controlled RFI-width test: deterministic full-time freq stripes of a fixed native-channel width
# over the clean sim baselines, at ~constant flag fraction. Reuses inject_rfi.py's schema so the
# output dataset drops straight into inpaint_infer / image_eval. Needs clean_baselines.h5 (from
# reextract.sh) for the run.
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
CLEAN_H5=${CLEAN_H5:?set CLEAN_H5=/scratch3/users/$USER/rfi/simulated/<run>/clean_baselines.h5}
OUT=${OUT:?set OUT=/path/to/dataset_wN.h5}
BAND_WIDTH=${BAND_WIDTH:?set BAND_WIDTH=<native channels>}
TARGET_FRAC=${TARGET_FRAC:-0.15}
SEED=${SEED:-42}

mkdir -p logs $(dirname $OUT)
singularity exec $ASTROPY python $ROOT/data_preparation/simulated/inject_rfi.py \
    --input "$CLEAN_H5" --output "$OUT" --band-width $BAND_WIDTH --target-frac $TARGET_FRAC --seed $SEED

echo "done -> $OUT"
