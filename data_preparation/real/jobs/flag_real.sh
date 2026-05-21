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

SRC_MS=/idia/data/public/1525469431/1525469431_sdp_l0.ms
WORKDIR=/scratch3/users/$USER/rfi/real
FLAGGED_MS=$WORKDIR/1525469431_flagged.ms
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
PATCHES_OUT=/scratch3/users/$USER/rfi/real_patches.h5
VIS_OUT=/scratch3/users/$USER/rfi/real_vis

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
    --config /usr/local/lib/python3.9/dist-packages/tricolour/conf/default.yaml \
    -fn ACT-CLJ2023.3-5535 \
    -nw 32 \
    -rc 10000 \
    -bc 24

echo "[3/4] $(date '+%H:%M:%S') extracting per-baseline patches"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/extract_ms.py \
    --ms $FLAGGED_MS \
    --output $PATCHES_OUT \
    --freq-min 900 \
    --freq-max 1650 \
    --field 0

echo "[4/4] $(date '+%H:%M:%S') visualising flagged data"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/visualisation/visualise_real.py \
    --ms $FLAGGED_MS \
    --output $VIS_OUT \
    --patches $PATCHES_OUT \
    --freq-min 900 \
    --freq-max 1650 \
    --field 0

echo "done $(date '+%H:%M:%S')"
echo "patches -> $PATCHES_OUT"
echo "plots   -> $VIS_OUT"
