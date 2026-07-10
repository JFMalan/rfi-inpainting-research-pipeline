#!/bin/bash
#SBATCH --job-name='rfi-oracle0'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/oracle0-%j-stdout.log
#SBATCH --error=logs/oracle0-%j-stderr.log

set -e

# Level-0 native-passthrough oracle (handover s4): verify the h5-unit -> MS-row map and write
# the EXACT native DATA at the hole pixels into ORACLE0_DATA (no h5 amp/phase, no resize round
# trip, no pol-collapse, no divisor). Then image clean vs flagged vs ORACLE0 with image_eval.
# If ORACLE0 != clean the bug is in the write-back row/channel/pol mechanics; if ORACLE0 == clean
# the loss is in the h5 representation (next test: phase-angle-resize, Suspect #1).
SIM=${SIM:-1}
MS=${MS:?set MS=/path/to/sim_clean.ms}
H5=${H5:?set H5=/path/to/dataset.h5}
MAX_UNITS=${MAX_UNITS:-}
OUTCOL=${OUTCOL:-ORACLE0_DATA}

ROOT=/users/$USER/rfi-inpainting-research-pipeline
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
mkdir -p logs

SIMARG=""; [ "$SIM" = "1" ] && SIMARG="--sim"
MU=""; [ -n "$MAX_UNITS" ] && MU="--max-units $MAX_UNITS"

echo "STAGE 1 (oracle write, ASTRO-PY3.10) node $(hostname)  ms=$MS h5=$H5 -> $OUTCOL"
singularity exec $ASTROPY python $ROOT/tests/oracle_level0.py \
    --ms "$MS" --h5 "$H5" --out-col "$OUTCOL" $SIMARG $MU

echo "STAGE 2 (image clean/flagged/$OUTCOL + compare)"
export SIM MS H5 MAX_UNITS
export INPCOL=$OUTCOL
export OUT=${OUT:-/idia/users/$USER/rfi/viz/oracle0}
bash $ROOT/evaluation/image_eval.sh

echo "done -> $OUT/image_comparison.png"
