#!/bin/bash
# Image test for the noise-free-target model (phase1_thr_paired). Writes back the recovered
# FINE STRUCTURE (noise_floor=none, no grain) and images Clean vs Flagged vs Inpaint vs DPSS
# to see whether filling the RFI holes with the recovered signal beats flagging them.
#
# Imaging base is the NOISE-FREE n0000 MS: the reference (Clean=DATA) is then perfect, so the
# only difference between Flagged and Inpaint is the RFI holes (flagging's lost sensitivity vs
# the fill's error) against a clean truth. Inference still conditions on the NOISY paired input.
# Run from repo root:  bash inference/jobs/noise_free_image_test.sh
set -e
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_thr_paired/best.pt}
H5=${H5:-/scratch3/users/$USER/rfi/simulated/runthr_paired/dataset.h5}
MS=${MS:-/scratch3/users/$USER/rfi/simulated/runthr_n0000/sim_clean.ms}
PREDS=/scratch3/users/$USER/rfi/preds_thr_paired.npz
VIZ=/idia/users/$USER/rfi/viz/noise_threshold/image_test
NODELIST=${NODELIST:-}   # e.g. gpu-005 to pin the GPU infer job
NODEARG=""; [ -n "$NODELIST" ] && NODEARG="--nodelist=$NODELIST"
mkdir -p logs "$VIZ"
for f in "$CKPT" "$H5" "$MS"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done

echo "[infer] GPU inference -> preds (fine structure, noise_floor=none)"
P=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS NOISE_FLOOR=none STEPS=50 \
    sbatch --parsable $NODEARG inference/jobs/inpaint_infer.sh)
echo "  -> infer $P"

echo "[write] write recovered signal into the noise-free MS (RESET_COL: context = clean DATA)"
W=$(env SIM=1 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS RESET_COL=1 \
    sbatch --parsable --dependency=afterok:$P inference/jobs/inpaint_writeback.sh)
echo "  -> write-back $W (waits on $P)"

echo "[image] wsclean: Clean vs Flagged vs Inpaint(fine) vs DPSS + delay spectrum"
I=$(env SIM=1 MS=$MS H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 DELAY=1 DPSS=1 DPSSFILL=1 \
    OUT=$VIZ \
    sbatch --parsable --dependency=afterok:$W evaluation/image_eval.sh)
echo "  -> image $I (waits on $W)"
echo ""
echo "watch: squeue -u \$USER"
echo "result: $VIZ/image_comparison.png + metrics.json (continuum RMSE-vs-clean: flagged vs inpaint vs DPSS)"
echo "        $VIZ/delay_spectrum.png (delay: now WITH a flagged/clean reference, unlike fill_check)"
