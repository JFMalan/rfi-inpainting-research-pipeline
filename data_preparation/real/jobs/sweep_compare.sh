#!/bin/bash
#SBATCH --job-name='rfi-sweep-compare'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/sweep-compare-%j-stdout.log
#SBATCH --error=logs/sweep-compare-%j-stderr.log

set -e

SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
FLAGGED_MS=/scratch3/users/$USER/rfi/real/1525469431_flagged.ms
CONFIGS=$SCRIPTS/data_preparation/real/jobs/sweep_configs.json
SWEEP_DIR=/scratch3/users/$USER/rfi/sweep
VIS_OUT=$SWEEP_DIR/comparison
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif

singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/visualisation/compare_sweep.py \
    --ms        $FLAGGED_MS \
    --configs   $CONFIGS \
    --sweep-dir $SWEEP_DIR \
    --output    $VIS_OUT \
    --field     0

echo "comparison plots -> $VIS_OUT"
