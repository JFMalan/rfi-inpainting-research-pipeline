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
MAXTSFLAG=${MAXTSFLAG:-0.95} # timestamp counts toward a good run unless >this fraction flagged; HIGHER = longer contiguous runs
MINRUN=${MINRUN:-64}         # shortest good run to keep
MAXFLAG=${MAXFLAG:-}         # per-tile keep threshold override (HIGHER = keep more-flagged tiles); empty = variant default 0.85
NOFORCE=${NOFORCE:-0}        # 1 = keep tricolour flags in persistent bands (location ceiling test); use a separate OUTDIR

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
OUTDIR=${OUTDIR:-/scratch3/users/$USER/rfi/real/variants}

mkdir -p $OUTDIR logs

ONLY_ARG=""; [ -n "$ONLY" ] && ONLY_ARG="--only $ONLY"
MAXFLAG_ARG=""; [ -n "$MAXFLAG" ] && MAXFLAG_ARG="--max-flag $MAXFLAG"
NOFORCE_ARG=""; [ "$NOFORCE" = "1" ] && NOFORCE_ARG="--no-force-persist"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/extract_variants.py \
    --ms $MS --out-dir $OUTDIR --column $COLUMN \
    --freq-min 900 --freq-max 1650 --img-size 512 \
    --max-ts-flag-frac $MAXTSFLAG --min-run $MINRUN $MAXFLAG_ARG $ONLY_ARG $NOFORCE_ARG

echo "done -> $OUTDIR"
ls -lh $OUTDIR
