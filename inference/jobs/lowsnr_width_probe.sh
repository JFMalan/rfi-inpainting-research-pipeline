#!/bin/bash
# Sky-brightness (per-visibility SNR) ablation at a fixed RFI stripe width. Per flux factor:
# reuse (or simulate) a dimmed sky with clean truth, inject BAND_WIDTH-channel stripes, inpaint,
# write back, image; compare inpaint vs flagged in the continuum AND (DELAY=1) the delay spectrum.
# Each level uses its own MS so writebacks never contend on a shared table.lock. Existing dim sims
# (rundim<F>: clean_baselines.h5 + sim_clean.ms) are reused, so only inject/infer/image re-run.
# Uses the bright-trained sim model -> OOD on dimmed data (a generalisation probe).
# Run from repo root:  bash inference/jobs/lowsnr_width_probe.sh
#   Part 2 usage:      BAND_WIDTH=<W*> DELAY=1 bash inference/jobs/lowsnr_width_probe.sh
set -e
FACTORS=${FACTORS:-"3 10 30"}
BAND_WIDTH=${BAND_WIDTH:-1}
DELAY=${DELAY:-0}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_tiled80ep/best.pt}
MAX_UNITS=${MAX_UNITS:-2000}
TARGET_FRAC=${TARGET_FRAC:-0.3}
NITER=${NITER:-3000}
SEED=${SEED:-100}
SIMROOT=/scratch3/users/$USER/rfi/simulated
VIZ=/idia/users/$USER/rfi/viz/lowsnr_width
mkdir -p logs "$VIZ"
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

for F in $FACTORS; do
    RID=dim$F
    SIMDIR=$SIMROOT/run$RID
    if [ -f "$SIMDIR/clean_baselines.h5" ] && [ -d "$SIMDIR/sim_clean.ms" ]; then
        echo "$RID: reusing existing sim ($SIMDIR)"
        SDEP=""
    else
        FMIN=$(python3 -c "print(0.1/$F)")
        FMAX=$(python3 -c "print(5.0/$F)")
        SJ=$(env RUN_ID=$RID SEED=$SEED GEN_RANDOM_SKY=1 FLUX_MIN=$FMIN FLUX_MAX=$FMAX \
             SYNTHESIS=2.4 NCHAN=1024 IMG_SIZE=512 NOISE_SCALE=1.0 TARGET_FRAC=0.15 \
             sbatch --parsable data_preparation/simulated/jobs/simulate.sh)
        SDEP="--dependency=afterok:$SJ"
        echo "$RID: simulate $SJ"
    fi
    H5=$SIMDIR/dataset_w${BAND_WIDTH}.h5
    PREDS=$SIMDIR/preds_w${BAND_WIDTH}.npz
    IJ=$(env CLEAN_H5=$SIMDIR/clean_baselines.h5 OUT=$H5 BAND_WIDTH=$BAND_WIDTH TARGET_FRAC=$TARGET_FRAC SEED=$SEED \
         sbatch --parsable $SDEP data_preparation/simulated/jobs/inject_width.sh)
    PJ=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS STEPS=50 MAX_UNITS=$MAX_UNITS \
         sbatch --parsable --dependency=afterok:$IJ inference/jobs/inpaint_infer.sh)
    WJ=$(env SIM=1 MS=$SIMDIR/sim_clean.ms H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS RESET_COL=1 \
         sbatch --parsable --dependency=afterok:$PJ inference/jobs/inpaint_writeback.sh)
    IMJ=$(env SIM=1 MS=$SIMDIR/sim_clean.ms H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 MEANFILL=1 \
         DPSSFILL=0 DPSS=$DELAY DELAY=$DELAY MAX_UNITS=$MAX_UNITS NITER=$NITER OUT=$VIZ/${RID}_w${BAND_WIDTH} \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    echo "$RID  flux/$F (~SNR $((90 / F)))  width=$BAND_WIDTH:  inject $IJ -> infer $PJ -> write $WJ -> image $IMJ"
done
echo ""
echo "results: $VIZ/dim<F>_w${BAND_WIDTH}/metrics.json (continuum) + delay_spectrum.png (DELAY=$DELAY)"