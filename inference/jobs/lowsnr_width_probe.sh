#!/bin/bash
# Width-1 continuum ablation across sky brightness (per-visibility SNR), to test whether
# thin-RFI inpaint-beats-flagging survives toward the real low-SNR regime. Per flux factor:
# simulate a dimmed sky (same seed, so only brightness changes) -> width-1 inject/infer/
# writeback/image, capped to MAX_UNITS baselines for a fair relative comparison. Each level
# uses its own MS so the writebacks never contend on a shared table.lock.
# Bright reference (SNR ~90) is the existing runtest width-1 result.
# NOTE: uses the bright-trained sim model; on dimmed data it is out-of-distribution, so this
# is a generalisation probe. Width-1 fills are near-local interpolation, the least-affected case.
# Run from repo root:  bash inference/jobs/lowsnr_width_probe.sh
set -e
FACTORS=${FACTORS:-"3 10 30"}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_tiled80ep/best.pt}
MAX_UNITS=${MAX_UNITS:-2000}
TARGET_FRAC=${TARGET_FRAC:-0.3}
SEED=${SEED:-100}
SIMROOT=/scratch3/users/$USER/rfi/simulated
VIZ=/idia/users/$USER/rfi/viz/lowsnr_width
mkdir -p logs "$VIZ"
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

for F in $FACTORS; do
    RID=dim$F
    SIMDIR=$SIMROOT/run$RID
    FMIN=$(python3 -c "print(0.1/$F)")
    FMAX=$(python3 -c "print(5.0/$F)")
    SJ=$(env RUN_ID=$RID SEED=$SEED GEN_RANDOM_SKY=1 FLUX_MIN=$FMIN FLUX_MAX=$FMAX \
         SYNTHESIS=2.4 NCHAN=1024 IMG_SIZE=512 NOISE_SCALE=1.0 TARGET_FRAC=0.15 \
         sbatch --parsable data_preparation/simulated/jobs/simulate.sh)
    H5=$SIMDIR/dataset_w1.h5
    PREDS=$SIMDIR/preds_w1.npz
    IJ=$(env CLEAN_H5=$SIMDIR/clean_baselines.h5 OUT=$H5 BAND_WIDTH=1 TARGET_FRAC=$TARGET_FRAC SEED=$SEED \
         sbatch --parsable --dependency=afterok:$SJ data_preparation/simulated/jobs/inject_width.sh)
    PJ=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS STEPS=50 MAX_UNITS=$MAX_UNITS \
         sbatch --parsable --dependency=afterok:$IJ inference/jobs/inpaint_infer.sh)
    WJ=$(env SIM=1 MS=$SIMDIR/sim_clean.ms H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS RESET_COL=1 \
         sbatch --parsable --dependency=afterok:$PJ inference/jobs/inpaint_writeback.sh)
    IMJ=$(env SIM=1 MS=$SIMDIR/sim_clean.ms H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 MEANFILL=1 \
         DPSSFILL=0 DPSS=0 DELAY=0 MAX_UNITS=$MAX_UNITS NITER=3000 OUT=$VIZ/$RID \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    echo "$RID  flux/$F (~SNR $((90 / F))):  sim $SJ -> inject $IJ -> infer $PJ -> write $WJ -> image $IMJ"
done
echo ""
echo "results: $VIZ/dim<F>/metrics.json  (compare inpainted vs flagged rmse_vs_clean per SNR)"
echo "bright reference (SNR ~90): /idia/users/$USER/rfi/viz/width_sweep/w1_cap/metrics.json"