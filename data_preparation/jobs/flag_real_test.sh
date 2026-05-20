#!/bin/bash
#SBATCH --job-name='rfi-flag-test'
#SBATCH --partition=Devel
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/flag-real-test-%j-stdout.log
#SBATCH --error=logs/flag-real-test-%j-stderr.log

set -e

SRC_MS=/idia/data/public/1525469431/1525469431_sdp_l0.ms
WORKDIR=/scratch3/users/$USER/rfi/real_test
SUBSET_MS=$WORKDIR/subset.ms
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
VIS_OUT=/scratch3/users/$USER/real_vis_test

mkdir -p $WORKDIR logs

OXKAT_SIF=$(ls /idia/software/containers/ | grep -i oxkat | head -1)
if [ -z "$OXKAT_SIF" ]; then
    echo "ERROR: no oxkat container found in /idia/software/containers/"
    ls /idia/software/containers/
    exit 1
fi
OXKAT=/idia/software/containers/$OXKAT_SIF
echo "using container: $OXKAT"

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
CASA=/idia/software/containers/casa-stable-v6.sif

rm -rf $SUBSET_MS

echo "[1/3] $(date '+%H:%M:%S') extracting scan 1 (~20 min) with CASA split"
singularity exec $CASA casa --nologger --log2term -c "
split(vis='$SRC_MS', outputvis='$SUBSET_MS', field='0', scan='1',
      datacolumn='data', keepflags=True)
"

echo "[2/3] $(date '+%H:%M:%S') running tricolour on subset"
singularity exec $OXKAT tricolour \
    $SUBSET_MS \
    -fs standard \
    -fn ACT-CLJ2023.3-5535 \
    -nw 8 \
    -rc 10000 \
    -bc 24

echo "[3/3] $(date '+%H:%M:%S') visualising"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/visualisation/visualise_real.py \
    --ms $SUBSET_MS \
    --output $VIS_OUT \
    --max-time 512

echo "done $(date '+%H:%M:%S')"
echo "plots -> $VIS_OUT"
