#!/bin/bash
# Sky-brightness (per-visibility SNR) ablation with REALISTIC RFI. Per flux factor: reuse an
# existing dimmed-sky sim (rundim<F>: clean truth + MS), inject stochastic RFI to TARGET_FRAC,
# inpaint, write back, image; compare inpaint vs flagged in the continuum AND (DELAY=1) the delay
# spectrum. Each level uses its own MS so writebacks never contend on a shared table.lock. Unlike
# lowsnr_width_probe.sh this uses realistic RFI geometry (the representative case), not full-time
# stripes, so it does not overstate the inpaint advantage. Uses the bright-trained sim model.
# Run from repo root:  DELAY=1 bash inference/jobs/lowsnr_real_probe.sh
set -e
FACTORS=${FACTORS:-"3 10 30"}
DELAY=${DELAY:-1}
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_tiled80ep/best.pt}
MAX_UNITS=${MAX_UNITS:-2000}
TARGET_FRAC=${TARGET_FRAC:-0.37}
NITER=${NITER:-3000}
SEED=${SEED:-100}
SIMROOT=/scratch3/users/$USER/rfi/simulated
VIZ=/idia/users/$USER/rfi/viz/lowsnr_real
mkdir -p logs "$VIZ"
[ -f "$CKPT" ] || { echo "ckpt not found: $CKPT"; exit 1; }

for F in $FACTORS; do
    RID=dim$F
    SIMDIR=$SIMROOT/run$RID
    if [ ! -f "$SIMDIR/clean_baselines.h5" ] || [ ! -d "$SIMDIR/sim_clean.ms" ]; then
        echo "$RID: MISSING sim ($SIMDIR) -- run the width probe first to build the dim sims"; exit 1
    fi
    echo "$RID: reusing existing sim ($SIMDIR)"
    H5=$SIMDIR/dataset_real.h5
    PREDS=$SIMDIR/preds_real.npz
    IJ=$(env CLEAN_H5=$SIMDIR/clean_baselines.h5 OUT=$H5 TARGET_FRAC=$TARGET_FRAC SEED=$SEED \
         sbatch --parsable data_preparation/simulated/jobs/inject_real.sh)
    PJ=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS STEPS=50 MAX_UNITS=$MAX_UNITS \
         sbatch --parsable --dependency=afterok:$IJ inference/jobs/inpaint_infer.sh)
    WJ=$(env SIM=1 MS=$SIMDIR/sim_clean.ms H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS RESET_COL=1 \
         sbatch --parsable --dependency=afterok:$PJ inference/jobs/inpaint_writeback.sh)
    IMJ=$(env SIM=1 MS=$SIMDIR/sim_clean.ms H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 MEANFILL=1 \
         DPSSFILL=0 DPSS=$DELAY DELAY=$DELAY MAX_UNITS=$MAX_UNITS NITER=$NITER OUT=$VIZ/${RID}_real \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    echo "$RID  flux/$F (~SNR $((90 / F))):  inject $IJ -> infer $PJ -> write $WJ -> image $IMJ"
done
echo ""
echo "results: $VIZ/dim<F>_real/metrics.json (continuum) + delay_spectrum.png (DELAY=$DELAY)"
