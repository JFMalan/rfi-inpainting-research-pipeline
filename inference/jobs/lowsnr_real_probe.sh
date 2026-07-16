#!/bin/bash
# Sky-brightness (per-visibility SNR) ablation with REALISTIC RFI. Per flux factor F: use a sim at
# 1/F of runtest's sky flux (runtest itself for F=1, rundim<F> otherwise; simulated on demand if
# missing), inject stochastic RFI to TARGET_FRAC, inpaint, write back, image; compare inpaint vs
# flagged in the continuum AND (DELAY=1) the delay spectrum. F=1 is the runtest anchor (~SNR 90);
# because runtest's MS is shared with the other sweeps it runs on a throwaway copy, removed after
# imaging. Dedicated dim MSes are used in place. Realistic RFI geometry (not full-time stripes) so
# the inpaint advantage is not overstated. Uses the bright-trained sim model.
# Run from repo root:  FACTORS="1 1.5 3 10 30" DELAY=1 bash inference/jobs/lowsnr_real_probe.sh
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
    SNR=$(python3 -c "print(round(90/$F))")
    if [ "$F" = "1" ]; then
        SIMDIR=$SIMROOT/runtest
        shared=1
    else
        SIMDIR=$SIMROOT/run$RID
        shared=0
    fi

    SDEP=""
    if [ ! -f "$SIMDIR/clean_baselines.h5" ] || [ ! -d "$SIMDIR/sim_clean.ms" ]; then
        if [ "$shared" = "1" ]; then echo "$RID: runtest sim missing ($SIMDIR) -- unexpected"; exit 1; fi
        FMIN=$(python3 -c "print(0.1/$F)")
        FMAX=$(python3 -c "print(5.0/$F)")
        SJ=$(env RUN_ID=$RID SEED=$SEED GEN_RANDOM_SKY=1 FLUX_MIN=$FMIN FLUX_MAX=$FMAX \
             SYNTHESIS=2.4 NCHAN=1024 IMG_SIZE=512 NOISE_SCALE=1.0 TARGET_FRAC=0.15 \
             sbatch --parsable data_preparation/simulated/jobs/simulate.sh)
        SDEP="--dependency=afterok:$SJ"
        echo "$RID: simulate $SJ (flux ${FMIN}-${FMAX} Jy)"
    else
        echo "$RID: reusing existing sim ($SIMDIR)"
    fi

    H5=$SIMDIR/dataset_real.h5
    PREDS=$SIMDIR/preds_real.npz
    IJ=$(env CLEAN_H5=$SIMDIR/clean_baselines.h5 OUT=$H5 TARGET_FRAC=$TARGET_FRAC SEED=$SEED \
         sbatch --parsable $SDEP data_preparation/simulated/jobs/inject_real.sh)
    PJ=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS STEPS=50 MAX_UNITS=$MAX_UNITS \
         sbatch --parsable --dependency=afterok:$IJ inference/jobs/inpaint_infer.sh)

    if [ "$shared" = "1" ]; then
        MSUSE=$SIMDIR/sim_clean_real_copy.ms
        CJ=$(env SRC=$SIMDIR/sim_clean.ms DST=$MSUSE sbatch --parsable inference/jobs/copy_ms.sh)
        WDEP="afterok:$PJ,afterok:$CJ"
    else
        MSUSE=$SIMDIR/sim_clean.ms
        CJ="-"
        WDEP="afterok:$PJ"
    fi
    WJ=$(env SIM=1 MS=$MSUSE H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS RESET_COL=1 \
         sbatch --parsable --dependency=$WDEP inference/jobs/inpaint_writeback.sh)
    IMJ=$(env SIM=1 MS=$MSUSE H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 MEANFILL=1 \
         DPSSFILL=0 DPSS=$DELAY DELAY=$DELAY MAX_UNITS=$MAX_UNITS NITER=$NITER OUT=$VIZ/${RID}_real \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    if [ "$shared" = "1" ]; then
        sbatch --parsable --dependency=afterok:$IMJ --job-name=rfi-rm-mscopy --partition=Main \
            --time=00:20:00 --mem=2GB --output=logs/rm-mscopy-%j.log --wrap="rm -rf $MSUSE" >/dev/null
    fi
    echo "$RID  flux/$F (~SNR $SNR):  inject $IJ -> infer $PJ -> copy $CJ -> write $WJ -> image $IMJ"
done
echo ""
echo "results: $VIZ/dim<F>_real/metrics.json (continuum) + delay_spectrum.png (DELAY=$DELAY)"
