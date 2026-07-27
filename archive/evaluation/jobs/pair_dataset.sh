#!/bin/bash
#SBATCH --job-name='rfi-pair-dataset'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/pair-dataset-%j-stdout.log
#SBATCH --error=logs/pair-dataset-%j-stderr.log

set -e
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
NOISY=${NOISY:?set NOISY (dataset.h5 at target noise)}
CLEAN=${CLEAN:?set CLEAN (dataset.h5 at NOISE_SCALE=0)}
OUT=${OUT:?set OUT (paired dataset.h5)}
mkdir -p logs $(dirname $OUT)

singularity exec $ASTROPY python $ROOT/archive/evaluation/make_paired_dataset.py \
    --noisy "$NOISY" --clean "$CLEAN" --out "$OUT"
echo "done -> $OUT"
