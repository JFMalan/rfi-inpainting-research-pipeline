#!/bin/bash
# REALISTIC-RFI flag-fraction sweep (bright runtest). The controlled full-time-stripe sweep
# (frac_sweep.sh) uses the most inpaint-favorable geometry and has inpaint winning the continuum at
# every fraction; this repeats the sweep with realistic stochastic RFI (bursty + intermittent +
# persistent bands + sweeps) to see whether inpaint beats flagging at ANY fraction on representative
# geometry, in the continuum AND (DELAY=1) the delay spectrum. Per fraction: inject realistic RFI to
# TARGET_FRAC, inpaint, write back, image. Serialized write-back+image on the shared runtest MS.
# Run from repo root:  bash inference/jobs/real_frac_sweep.sh
set -e
FRACS=${FRACS:-"0.15 0.3 0.45 0.6"}
RUN=${RUN:-runtest}
DELAY=${DELAY:-1}
SEED=${SEED:-100}
SIMDIR=/scratch3/users/$USER/rfi/simulated/$RUN
CLEAN_H5=${CLEAN_H5:-$SIMDIR/clean_baselines.h5}
MS=${MS:-$SIMDIR/sim_clean.ms}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_tiled80ep/best.pt}
MAX_UNITS=${MAX_UNITS:-2000}
NITER=${NITER:-3000}
DATADIR=${DATADIR:-/scratch3/users/$USER/rfi/real_frac_sweep}
VIZ=/idia/users/$USER/rfi/viz/real_frac_sweep
mkdir -p logs "$DATADIR" "$VIZ"
for f in "$CLEAN_H5" "$MS" "$CKPT"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done

echo "=== phase 1: inject realistic RFI + infer per fraction (parallel) ==="
declare -A PJID
for FR in $FRACS; do
    TAG=f${FR/./p}
    H5=$DATADIR/dataset_${TAG}.h5
    IJ=$(env CLEAN_H5=$CLEAN_H5 OUT=$H5 TARGET_FRAC=$FR SEED=$SEED \
         sbatch --parsable data_preparation/simulated/jobs/inject_real.sh)
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
    IMJ=$(env SIM=1 MS=$MS H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 DELAY=$DELAY DPSS=$DELAY MEANFILL=1 DPSSFILL=0 \
         MAX_UNITS=$MAX_UNITS NITER=$NITER OUT=$VIZ/$TAG \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    echo "  frac=$FR  write $WJ -> image $IMJ"
    prev=$IMJ
done
echo ""
echo "result: $VIZ/f<frac>/metrics.json  (inpaint vs flag, continuum + delay, realistic RFI)"
