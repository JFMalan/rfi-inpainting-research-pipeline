#!/bin/bash
#SBATCH --job-name='rfi-recoverable-real'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/recoverable-real-%j-stdout.log
#SBATCH --error=logs/recoverable-real-%j-stderr.log

set -e

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
REAL=${REAL:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
OUT=${OUT:-/scratch3/users/$USER/rfi/vis-layers-real}

# how much of REAL amplitude is recoverable structure vs irreducible noise, with the
# 2D low-pass split. 8 example layer figures + mean per-layer autocorr summary.
singularity exec $ASTROPY python $SCRIPTS/archive/data_preparation/simulated/visualisation/decompose_layers.py \
    --data $REAL --n 8 --sigma 2 --output $OUT
