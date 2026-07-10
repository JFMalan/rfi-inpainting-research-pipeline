#!/bin/bash
#SBATCH --job-name='rfi-recover-real'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/recover-real-%j-stdout.log
#SBATCH --error=logs/recover-real-%j-stderr.log

set -e

DATA=${DATA:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

singularity exec $ASTROPY python $SCRIPTS/archive/data_preparation/real/recover_real.py --data $DATA --n 200
