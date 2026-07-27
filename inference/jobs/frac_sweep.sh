#!/bin/bash
# Flag-FRACTION sweep at fixed RFI stripe width (bright runtest). The width sweep at fixed 30%
# showed inpaint beats flagging in the continuum for all widths to 64 (fill error saturates while
# flagging's sensitivity loss stays fixed) - so the real crossover lever is RFI fraction, not width.
# This sweeps TARGET_FRAC at fixed WIDTH to find the fraction where flagging retakes the continuum
# lead, mapping onto the real 48-85%-flagged regime. Per fraction: inject WIDTH-channel stripes to
# TARGET_FRAC, inpaint, write back, image; compare inpaint vs flagged. Serialized write-back+image
# on the shared runtest MS (must run AFTER the width sweep frees that MS).
# Run from repo root:  bash inference/jobs/frac_sweep.sh
set -e
FRACS=${FRACS:-"0.2 0.35 0.5 0.65 0.8"}
WIDTH=${WIDTH:-32}
RUN=${RUN:-runtest}
SIMDIR=/scratch3/users/$USER/rfi/simulated/$RUN
CLEAN_H5=${CLEAN_H5:-$SIMDIR/clean_baselines.h5}
MS=${MS:-$SIMDIR/sim_clean.ms}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_tiled80ep/best.pt}
MAX_UNITS=${MAX_UNITS:-2000}
NITER=${NITER:-3000}
DATADIR=${DATADIR:-/scratch3/users/$USER/rfi/frac_sweep}
VIZ=/idia/users/$USER/rfi/viz/frac_sweep
mkdir -p logs "$DATADIR" "$VIZ"
for f in "$CLEAN_H5" "$MS" "$CKPT"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done

echo "=== phase 1: inject + infer per fraction (parallel), width=$WIDTH ==="
declare -A PJID
for FR in $FRACS; do
    TAG=f${FR/./p}
    H5=$DATADIR/dataset_${TAG}.h5
    IJ=$(env CLEAN_H5=$CLEAN_H5 OUT=$H5 BAND_WIDTH=$WIDTH TARGET_FRAC=$FR SEED=100 \
         sbatch --parsable data_preparation/simulated/jobs/inject_width.sh)
    PJ=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$DATADIR/preds_${TAG}.npz STEPS=50 \
         MAX_UNITS=$MAX_UNITS \
         sbatch --parsable --dependency=afterok:$IJ inference/jobs/inpaint_infer.sh)
    PJID[$FR]=$PJ
    echo "  frac=$FR  inject $IJ -> infer $PJ"
done

echo "=== phase 2: write-back + image per fraction (serial on the sim MS) ==="
prev=""
for FR in $FRACS; do
    TAG=f${FR/./p}
    H5=$DATADIR/dataset_${TAG}.h5
    deps="afterok:${PJID[$FR]}"; [ -n "$prev" ] && deps="$deps,afterok:$prev"
    WJ=$(env SIM=1 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$DATADIR/preds_${TAG}.npz RESET_COL=1 \
         sbatch --parsable --dependency=$deps inference/jobs/inpaint_writeback.sh)
    IMJ=$(env SIM=1 MS=$MS H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 DELAY=0 DPSS=0 MEANFILL=1 DPSSFILL=0 \
         MAX_UNITS=$MAX_UNITS NITER=$NITER OUT=$VIZ/$TAG \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    echo "  frac=$FR  write $WJ -> image $IMJ"
    prev=$IMJ
done
echo ""
echo "result: $VIZ/f<frac>/metrics.json  (inpaint vs flag rmse_vs_clean per RFI fraction, width=$WIDTH)"