#!/bin/bash
# Controlled RFI-width sweep in the SIMULATED domain. For each band width: inject deterministic
# stripes over the clean test baselines, inpaint, image with wsclean, and compare the continuum
# to the clean truth and the flagged version. Thin RFI should inpaint near-perfectly (image beats
# flagging); the sweep finds the width where accumulated fill error makes inpaint worse than flag.
# Run from repo root:  bash inference/jobs/rfi_width_sweep.sh
#
# inject + infer run in parallel across widths (MS-independent); write-back + imaging serialize on
# the shared sim MS (afterok chain). Uses the SIM model on in-domain sim data.
set -e
WIDTHS=${WIDTHS:-"2 4 8 16 32 64"}   # width 1 already done as w1_cap; find where inpaint stops beating flag
RUN=${RUN:-runtest}
SIMDIR=/scratch3/users/$USER/rfi/simulated/$RUN
CLEAN_H5=${CLEAN_H5:-$SIMDIR/clean_baselines.h5}
MS=${MS:-$SIMDIR/sim_clean.ms}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_tiled80ep/best.pt}
TARGET_FRAC=${TARGET_FRAC:-0.3}
MAX_UNITS=${MAX_UNITS:-2000}         # cap baselines for a fair, faster relative comparison (matches w1_cap)
NITER=${NITER:-3000}
DATADIR=${DATADIR:-/scratch3/users/$USER/rfi/width_sweep}
VIZ=/idia/users/$USER/rfi/viz/width_sweep
mkdir -p logs "$DATADIR" "$VIZ"

for f in "$CLEAN_H5" "$MS" "$CKPT"; do
    [ -e "$f" ] || { echo "missing: $f  (clean_baselines.h5 comes from reextract.sh on $RUN)"; exit 1; }
done

SKIP_PHASE1=${SKIP_PHASE1:-0}   # 1 = reuse existing datasets/preds on disk; only re-run write-back+image+summary
declare -A PJID
if [ "$SKIP_PHASE1" = "1" ]; then
    echo "=== phase 1 SKIPPED: reusing $DATADIR/dataset_w*.h5 + preds_w*.npz ==="
    for W in $WIDTHS; do
        [ -f "$DATADIR/preds_w$W.npz" ] || { echo "missing preds_w$W.npz -- run phase 1 first"; exit 1; }
    done
else
    echo "=== phase 1: inject + infer per width (parallel) ==="
    for W in $WIDTHS; do
        H5=$DATADIR/dataset_w$W.h5
        IJ=$(env CLEAN_H5=$CLEAN_H5 OUT=$H5 BAND_WIDTH=$W TARGET_FRAC=$TARGET_FRAC \
             sbatch --parsable data_preparation/simulated/jobs/inject_width.sh)
        PJ=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$DATADIR/preds_w$W.npz STEPS=50 \
             MAX_UNITS=$MAX_UNITS \
             sbatch --parsable --dependency=afterok:$IJ inference/jobs/inpaint_infer.sh)
        PJID[$W]=$PJ
        echo "  w=$W  inject $IJ -> infer $PJ"
    done
fi

echo "=== phase 2: write-back + image per width (serial on the sim MS) ==="
prev=""
for W in $WIDTHS; do
    H5=$DATADIR/dataset_w$W.h5
    deps=""; [ -n "${PJID[$W]}" ] && deps="afterok:${PJID[$W]}"
    [ -n "$prev" ] && deps="${deps:+$deps,}afterok:$prev"
    DEPARG=""; [ -n "$deps" ] && DEPARG="--dependency=$deps"
    WJ=$(env SIM=1 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$DATADIR/preds_w$W.npz RESET_COL=1 \
         sbatch --parsable $DEPARG inference/jobs/inpaint_writeback.sh)
    IMJ=$(env SIM=1 MS=$MS H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 DELAY=0 DPSS=0 MEANFILL=1 DPSSFILL=0 \
         MAX_UNITS=$MAX_UNITS NITER=$NITER OUT=$VIZ/w$W \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    echo "  w=$W  write $WJ -> image $IMJ"
    prev=$IMJ
done

SUM=$(env SWEEP_ROOT=$VIZ WIDTHS="$WIDTHS" OUT=$VIZ/width_sweep_summary.png \
      sbatch --parsable --dependency=afterok:$prev evaluation/jobs/width_sweep_summary.sh)
echo "=== summary plot: $SUM (afterok $prev) ==="
echo ""
echo "watch: squeue -u \$USER"
echo "result: $VIZ/width_sweep_summary.png  (+ per-width $VIZ/w<N>/image_comparison.png, metrics.json)"
