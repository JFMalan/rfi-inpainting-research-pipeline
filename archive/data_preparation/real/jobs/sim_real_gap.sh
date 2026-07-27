#!/bin/bash
#SBATCH --job-name='rfi-sim-real-gap'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/sim-real-gap-%j-stdout.log
#SBATCH --error=logs/sim-real-gap-%j-stderr.log

set -e

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
OUT=${OUT:-$SCRIPTS/data_preparation/real/vis-real/sim_real_gap}

singularity exec $ASTROPY python $SCRIPTS/archive/data_preparation/real/sim_real_gap.py \
    --sim "/scratch3/users/$USER/rfi/simulated/run*/dataset.h5" \
    --real /scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5 \
    --n ${N:-300} \
    --output $OUT
