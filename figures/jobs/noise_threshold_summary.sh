#!/bin/bash
#SBATCH --job-name='rfi-noise-thr-summary'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/noise-thr-summary-%j-stdout.log
#SBATCH --error=logs/noise-thr-summary-%j-stderr.log

set -e
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
RUNS_ROOT=${RUNS_ROOT:-/idia/users/$USER/rfi/runs}
SCALES=${SCALES:-"1.0 0.5 0.25 0.125"}
CLEAN_H5=${CLEAN_H5:-}
OUT=${OUT:-/idia/users/$USER/rfi/viz/noise_threshold/noise_threshold.png}

singularity exec $ASTROPY python $ROOT/figures/plot_noise_threshold.py \
    --runs-root "$RUNS_ROOT" --scales "$SCALES" --clean-h5 "$CLEAN_H5" --out "$OUT"
echo "done -> $OUT"
