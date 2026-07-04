#!/bin/bash
# Overnight: down-weight the inpainted pixels and re-image the continuum (WEIGHT_FRAC sweep), then
# confirm the end-to-end delay-space win on the written INPAINTED_DATA (subsampled so it fits).
# All steps edit/read the SAME MS, so they run as one afterok chain (no overlap). Reuses the preds
# already on disk (no GPU inference). Run from repo root:  bash inference/jobs/downweight_delay_queue.sh
set -e
MS=${MS:-/scratch3/users/$USER/rfi/real/inpaint_target.ms}
H5=${H5:-/scratch3/users/$USER/rfi/real/variants/v6_native512.h5}
PREDS=${PREDS:-/scratch3/users/$USER/rfi/inpaint_preds_selective.npz}
VIZ=/idia/users/$USER/rfi/viz

if [ ! -f "$PREDS" ]; then echo "preds missing: $PREDS (run the selective inpaint infer first)"; exit 1; fi

prev=""
dep() { [ -n "$prev" ] && printf -- "--dependency=afterok:%s" "$prev"; }

for f in 0.2 0.3 0.5; do
    echo "[wf=$f] down-weight write-back (non-persistent holes -> $f x weight)"
    W=$(env SIM=0 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS KEEP_PERSIST=1 WEIGHT_FRAC=$f \
        sbatch --parsable $(dep) inference/jobs/inpaint_writeback.sh)
    echo "  -> write $W"; prev=$W
    echo "[wf=$f] continuum image (Flagged-everything vs Selective @ weight $f; delay skipped)"
    I=$(env SIM=0 MS=$MS H5=$H5 KEEP_PERSIST=1 DO_INPAINT=1 DELAY=0 DPSS=0 OUT=$VIZ/image_wf${f} \
        sbatch --parsable --dependency=afterok:$prev evaluation/image_eval.sh)
    echo "  -> image $I"; prev=$I
done

echo "[delay] end-to-end delay confirmation on written INPAINTED_DATA (subsampled 800 units, +DPSS)"
D=$(env MS=$MS H5=$H5 INPCOL=INPAINTED_DATA MAX_UNITS=800 OUT=$VIZ/delay_confirm \
    sbatch --parsable --dependency=afterok:$prev evaluation/jobs/delay_confirm.sh)
echo "  -> delay $D"

echo ""
echo "chain submitted (serial on the MS). watch: squeue -u \$USER"
echo "results: $VIZ/image_wf{0.2,0.3,0.5}/image_comparison.png ; $VIZ/delay_confirm/delay_spectrum.png"
echo "read continuum: grep -H 'off-src RMS\\|inpainted\\|flagged\\|dyn' logs/image-eval-*-stdout.log | tail"
