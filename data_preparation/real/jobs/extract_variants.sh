#!/bin/bash
#SBATCH --job-name='rfi-extract-variants'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/extract-variants-%j-stdout.log
#SBATCH --error=logs/extract-variants-%j-stderr.log

set -e

MS=${MS:-/idia/projects/astro-cirg/data_for_rfi/1570802018_sdp_l0-J2018_5539-corr.ms}
COLUMN=${COLUMN:-DATA}
ONLY=${ONLY:-v6_native512}   # tiled variant matching the sim extractor; set ONLY= to build the legacy v1-v5

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
OUTDIR=/scratch3/users/$USER/rfi/real/variants

mkdir -p $OUTDIR logs

ONLY_ARG=""; [ -n "$ONLY" ] && ONLY_ARG="--only $ONLY"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/extract_variants.py \
    --ms $MS --out-dir $OUTDIR --column $COLUMN \
    --freq-min 900 --freq-max 1650 --img-size 512 $ONLY_ARG

echo "done -> $OUTDIR"
ls -lh $OUTDIR
