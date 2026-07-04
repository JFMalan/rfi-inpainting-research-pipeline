#!/bin/bash
#SBATCH --job-name='rfi-delay-confirm'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/delay-confirm-%j-stdout.log
#SBATCH --error=logs/delay-confirm-%j-stderr.log

set -e

# End-to-end delay-space confirmation on the WRITTEN INPAINTED_DATA (actual RFI holes, not fake
# holes). Standalone so it gets its own walltime; image_eval's delay step timed out. MAX_UNITS
# subsamples baselines so it fits the wall. INPAINTED_DATA is weight-independent here, so this is
# valid regardless of the WEIGHT_FRAC used at write time.
MS=${MS:-/scratch3/users/$USER/rfi/real/inpaint_target.ms}
H5=${H5:-/scratch3/users/$USER/rfi/real/variants/v6_native512.h5}
INPCOL=${INPCOL:-INPAINTED_DATA}
MAX_UNITS=${MAX_UNITS:-800}
OUT=${OUT:-/idia/users/$USER/rfi/viz/delay_confirm}

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
mkdir -p logs $(dirname $OUT)

singularity exec $ASTROPY python $ROOT/evaluation/delay_spectrum.py \
    --ms "$MS" --h5 "$H5" --inp-col "$INPCOL" --dpss --max-units $MAX_UNITS \
    --out $OUT/delay_spectrum.png

echo "done -> $OUT/delay_spectrum.png"
