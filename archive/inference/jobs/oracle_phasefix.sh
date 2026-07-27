#!/bin/bash
#SBATCH --job-name='rfi-oracle-pfix'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/oracle-pfix-%j-stdout.log
#SBATCH --error=logs/oracle-pfix-%j-stderr.log

set -e

# Phase-fix oracle: identical to inpaint_infer --oracle except phase is reconstructed by resizing
# cos/sin (computed from the NATIVE angle) instead of resizing the wrapped angle. Same true amp,
# so it isolates the phase-angle-resize fix (Suspect #1) at the image level. Image clean vs flagged
# vs ORACLE_PFIX. If PFIX beats/ties flagged, the fix is worth a re-extract + retrain.
SIM=${SIM:-1}
MS=${MS:?set MS=/path/to/sim_clean.ms}
H5=${H5:?set H5=/path/to/dataset.h5}
MAX_UNITS=${MAX_UNITS:-}
# SMOOTH_AMP=1 writes smooth_component(clean) amplitude (the DECOMPOSE-model ceiling: best a
# perfect smooth/decompose fill can do) instead of the true full clean amplitude.
SMOOTH_AMP=${SMOOTH_AMP:-0}
if [ "$SMOOTH_AMP" = "1" ]; then OUTCOL=${OUTCOL:-ORACLE_SMOOTH_DATA}; else OUTCOL=${OUTCOL:-ORACLE_PFIX_DATA}; fi

ROOT=/users/$USER/rfi-inpainting-research-pipeline
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
PREDS=${PREDS:-/scratch3/users/$USER/rfi/oracle_pfix_preds_${SLURM_JOB_ID}.npz}
mkdir -p logs

SIMARG=""; [ "$SIM" = "1" ] && SIMARG="--sim"
SAARG=""; [ "$SMOOTH_AMP" = "1" ] && SAARG="--smooth-amp"
MU=""; [ -n "$MAX_UNITS" ] && MU="--max-units $MAX_UNITS"

echo "STAGE A (build phase-fix preds, ASTRO-PY3.10) node $(hostname)  ms=$MS h5=$H5 smooth_amp=$SMOOTH_AMP -> $PREDS"
singularity exec $ASTROPY python $ROOT/archive/inference/oracle_phasefix.py \
    --ms "$MS" --h5 "$H5" --out-preds "$PREDS" $SIMARG $SAARG $MU

echo "STAGE B (write-back $PREDS -> $OUTCOL)"
singularity exec $ASTROPY python $ROOT/inference/inpaint_write.py \
    --ms "$MS" --h5 "$H5" --preds "$PREDS" --out-col "$OUTCOL" $SIMARG

echo "STAGE C (image clean/flagged/$OUTCOL + compare)"
export SIM MS H5 MAX_UNITS
export INPCOL=$OUTCOL
if [ "$SMOOTH_AMP" = "1" ]; then export OUT=${OUT:-/idia/users/$USER/rfi/viz/oracle_smooth}; else export OUT=${OUT:-/idia/users/$USER/rfi/viz/oracle_pfix}; fi
bash $ROOT/evaluation/image_eval.sh

echo "done -> $OUT/image_comparison.png"
