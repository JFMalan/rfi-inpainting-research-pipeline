#!/bin/bash
#SBATCH --job-name='rfi-inpaint-write'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/inpaint-write-%j-stdout.log
#SBATCH --error=logs/inpaint-write-%j-stderr.log

set -e

# Stage 2 only: CPU write-back of a preds .npz (from inpaint_infer.sh) into the MS. Runs on Main
# where the 128GB it needs is easy to get, off the contended GPU nodes. Same PREDS/MS/H5/OUTCOL
# as the infer job.
SIM=${SIM:-0}
OUTCOL=${OUTCOL:-INPAINTED_DATA}
UNFLAG=${UNFLAG:-0}
KEEP_PERSIST=${KEEP_PERSIST:-0}
WEIGHT_FRAC=${WEIGHT_FRAC:-}
RESET_COL=${RESET_COL:-0}   # 1 = re-copy DATA into the out-col first (needed when reusing one column across a sweep)

ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
MS=${MS:?set MS=/path/to/flagged.ms}
H5=${H5:?set H5=/path/to/extracted_dataset.h5}
PREDS=${PREDS:?set PREDS=/path/to/inpaint_preds.npz (same as inpaint_infer.sh)}

if [ ! -f "$PREDS" ]; then echo "preds not found: $PREDS (did inpaint_infer.sh finish?)"; exit 1; fi
mkdir -p logs

WRITE_EXTRA=""
[ "$SIM" = "1" ] && WRITE_EXTRA="$WRITE_EXTRA --sim"
[ "$UNFLAG" = "1" ] && WRITE_EXTRA="$WRITE_EXTRA --unflag"
[ "$KEEP_PERSIST" = "1" ] && WRITE_EXTRA="$WRITE_EXTRA --keep-persist-flagged"
[ "$RESET_COL" = "1" ] && WRITE_EXTRA="$WRITE_EXTRA --reset-col"
[ -n "$WEIGHT_FRAC" ] && WRITE_EXTRA="$WRITE_EXTRA --weight-frac $WEIGHT_FRAC"

echo "CPU write-back  $PREDS -> $OUTCOL in $MS"
singularity exec $ASTROPY python $ROOT/inference/inpaint_write.py \
    --ms "$MS" --h5 "$H5" --preds "$PREDS" --out-col "$OUTCOL" $WRITE_EXTRA

echo "done -> $OUTCOL in $MS"
