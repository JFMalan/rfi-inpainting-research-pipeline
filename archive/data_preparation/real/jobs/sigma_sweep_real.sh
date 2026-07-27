#!/bin/bash
#SBATCH --job-name='rfi-sigma-sweep'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/sigma-sweep-%j-stdout.log
#SBATCH --error=logs/sigma-sweep-%j-stderr.log

set -e

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
REAL=${REAL:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
OUTROOT=${OUTROOT:-/scratch3/users/$USER/rfi/sigma-sweep-real}

# find the low-pass cutoff that maximises recoverable smooth structure while driving the
# residual toward white (grain ac -> 0). lower sigma = sharper cut = keeps more structure.
for SIG in 1.0 1.5 2.0 3.0 4.0; do
    echo ""
    echo "############### SIGMA = $SIG ###############"
    singularity exec $ASTROPY python $SCRIPTS/archive/data_preparation/simulated/visualisation/decompose_layers.py \
        --data $REAL --n 8 --sigma $SIG --output $OUTROOT/sig${SIG} | grep -E "MEAN|^patch"
done
echo ""
echo "pick the sigma with highest smooth std AND lowest grain ac (most recoverable, residual most white)"
