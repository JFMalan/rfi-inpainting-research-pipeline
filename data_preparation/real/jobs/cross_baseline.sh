#!/bin/bash
#SBATCH --job-name='rfi-xbl-recover'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/xbl-recover-%j-stdout.log
#SBATCH --error=logs/xbl-recover-%j-stderr.log

set -e

MS=${MS:-/idia/projects/astro-cirg/data_for_rfi/1570802018_sdp_l0-J2018_5539-corr.ms}
COLUMN=${COLUMN:-DATA}
FMIN=${FMIN:-1300}
FMAX=${FMAX:-1370}
NCHAN=${NCHAN:-32}

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
OUT=/scratch3/users/$USER/rfi/real/xbl_recover.npz
mkdir -p logs $(dirname $OUT)

echo "MS=$MS COLUMN=$COLUMN window=${FMIN}-${FMAX} MHz"

singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/cross_baseline_recoverability.py \
    --ms $MS --column $COLUMN --freq-min $FMIN --freq-max $FMAX --n-chan $NCHAN \
    --out $OUT

echo "done -> $OUT"
