#!/bin/bash
#SBATCH --job-name='rfi-flag-real'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=10:00:00
#SBATCH --output=logs/flag-real-%j-stdout.log
#SBATCH --error=logs/flag-real-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

SRC_MS=${SRC_MS:-/idia/data/public/1525469431/1525469431_sdp_l0.ms}
FIELD_NAME=${FIELD_NAME:-ACT-CLJ2023.3-5535}   # tricolour -fn, per-observation metadata
FIELD=${FIELD:-0}
FREQ_MIN=${FREQ_MIN:-900}
FREQ_MAX=${FREQ_MAX:-1650}
MAX_BL_FLAG=${MAX_BL_FLAG:-}
SMOOTH_BINS=${SMOOTH_BINS:-}
WORKDIR=${WORKDIR:-/scratch3/users/$USER/rfi/real}
FLAGGED_MS=${FLAGGED_MS:-$WORKDIR/$(basename ${SRC_MS%.ms})_flagged.ms}
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
PATCHES_OUT=${PATCHES_OUT:-/scratch3/users/$USER/rfi/real_patches.h5}
VIS_OUT=${VIS_OUT:-/scratch3/users/$USER/rfi/real_vis}

mkdir -p $WORKDIR logs

OXKAT_SIF=$(ls /idia/software/containers/ | grep -i oxkat | head -1)
if [ -z "$OXKAT_SIF" ]; then
    echo "ERROR: no oxkat container found in /idia/software/containers/"
    echo "available containers:"
    ls /idia/software/containers/
    exit 1
fi
OXKAT=/idia/software/containers/$OXKAT_SIF
echo "using container: $OXKAT"

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif

echo "[1/4] $(date '+%H:%M:%S') copying MS to scratch (read-only source)"
cp -r $SRC_MS $FLAGGED_MS

echo "[2/4] $(date '+%H:%M:%S') running tricolour"
singularity exec $OXKAT tricolour \
    $FLAGGED_MS \
    --config $SCRIPTS/data_preparation/real/tricolour-flagging.yaml \
    -fn $FIELD_NAME \
    -nw 32 \
    -rc 10000 \
    -bc 24 \
    -dpm

EXTRACT_EXTRA=""
if [ -n "$MAX_BL_FLAG" ]; then EXTRACT_EXTRA="$EXTRACT_EXTRA --max-bl-flag-frac $MAX_BL_FLAG"; fi
if [ -n "$SMOOTH_BINS" ]; then EXTRACT_EXTRA="$EXTRACT_EXTRA --smooth-bins $SMOOTH_BINS"; fi

echo "[3/4] $(date '+%H:%M:%S') extracting per-baseline patches"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/extract_ms.py \
    --ms $FLAGGED_MS \
    --output $PATCHES_OUT \
    --freq-min $FREQ_MIN \
    --freq-max $FREQ_MAX \
    --field $FIELD $EXTRACT_EXTRA

echo "[4/4] $(date '+%H:%M:%S') visualising flagged data"
singularity exec $ASTROPY python $SCRIPTS/figures/visualise_real.py \
    --ms $FLAGGED_MS \
    --output $VIS_OUT \
    --freq-min $FREQ_MIN \
    --freq-max $FREQ_MAX \
    --field $FIELD

echo "done $(date '+%H:%M:%S')"
echo "patches -> $PATCHES_OUT"
echo "plots   -> $VIS_OUT"
