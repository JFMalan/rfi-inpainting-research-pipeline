#!/bin/bash
#SBATCH --job-name='rfi-extract-real'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/extract-real-%j-stdout.log
#SBATCH --error=logs/extract-real-%j-stderr.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

MS=${MS:-/idia/projects/astro-cirg/data_for_rfi/1570802018_sdp_l0-J2018_5539-corr.ms}
COLUMN=${COLUMN:-DATA}
MAX_BL_FLAG=${MAX_BL_FLAG:-0.5}
IMG_SIZE=${IMG_SIZE:-512}

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
OUTDIR=/scratch3/users/$USER/rfi/real
OUT_H5=$OUTDIR/dataset.h5

mkdir -p $OUTDIR logs

echo "MS=$MS  COLUMN=$COLUMN  MAX_BL_FLAG=$MAX_BL_FLAG  IMG_SIZE=$IMG_SIZE"

singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/extract_ms.py \
    --ms               $MS \
    --output           $OUT_H5 \
    --column           $COLUMN \
    --freq-min         900 \
    --freq-max         1650 \
    --img-size         $IMG_SIZE \
    --max-bl-flag-frac $MAX_BL_FLAG

echo "done -> $OUT_H5"
