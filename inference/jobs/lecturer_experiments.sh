#!/bin/bash
# One script for the supervisor's requests, in the SIMULATED domain (clean truth available):
#   1. Thermal-noise generalisation: same sky, several noise levels (incl. noise-free and OOD),
#      eval the EXISTING model -> does accuracy hold at noise levels it never trained on?
#   2. Blending ablation: model fill WITH vs WITHOUT the feathered tile blend (one component
#      toggled, Massoud-style ablation), imaged and compared to clean + flagged + DPSS.
# Run from repo root:  bash inference/jobs/lecturer_experiments.sh
#
# Per noise level: simulate(same sky, NOISE_SCALE=S) -> infer -> write-back -> image_eval
# (clean/flagged/model/DPSS vs clean). Levels use separate MSes so they run IN PARALLEL.
set -e
NOISE_SCALES=${NOISE_SCALES:-"0 1 2 4"}   # 0=noise-free, 1=physical MeerKAT SEFD (in-dist), 2/4=OOD-high
SKY_MODEL=${SKY_MODEL:-sky_model_bright.txt}   # SAME sky across levels so only noise varies
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/phase1_all_tiled80ep/best.pt}   # sim full-amp model (in-domain)
SIMROOT=/scratch3/users/$USER/rfi/simulated
VIZ=/idia/users/$USER/rfi/viz/lecturer
mkdir -p logs "$VIZ"
[ -f "$CKPT" ] || { echo "sim checkpoint not found: $CKPT"; exit 1; }

declare -A IMJID
echo "=== (1) NOISE GENERALISATION: sky=$SKY_MODEL  scales=$NOISE_SCALES  model=$(basename $(dirname $CKPT)) ==="
for S in $NOISE_SCALES; do
    RID=lec_n$S; SIMD=$SIMROOT/run$RID; H5=$SIMD/dataset.h5; MS=$SIMD/sim_clean.ms
    PREDS=$SIMROOT/preds_$RID.npz
    SJ=$(env RUN_ID=$RID NOISE_SCALE=$S SKY_MODEL=$SKY_MODEL GEN_RANDOM_SKY=0 \
         sbatch --parsable data_preparation/simulated/jobs/simulate.sh)
    PJ=$(env SIM=1 SMOOTH=0 CKPT=$CKPT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS STEPS=50 \
         sbatch --parsable --dependency=afterok:$SJ inference/jobs/inpaint_infer.sh)
    WJ=$(env SIM=1 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS RESET_COL=1 \
         sbatch --parsable --dependency=afterok:$PJ inference/jobs/inpaint_writeback.sh)
    IJ=$(env SIM=1 MS=$MS H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 DELAY=0 DPSS=0 DPSSFILL=1 MEANFILL=0 \
         OUT=$VIZ/n$S \
         sbatch --parsable --dependency=afterok:$WJ evaluation/image_eval.sh)
    IMJID[$S]=$IJ
    echo "  noise=$S: sim $SJ -> infer $PJ -> write $WJ -> image $IJ  (-> $VIZ/n$S)"
done

alljobs=$(IFS=,; echo "${IMJID[*]}")
SUM=$(env SWEEP_ROOT=$VIZ WIDTHS="$NOISE_SCALES" PREFIX=n LOGX=0 \
      XLABEL="thermal noise (x MeerKAT SEFD)" OUT=$VIZ/noise_generalisation.png \
      sbatch --parsable --dependency=afterok:$alljobs evaluation/jobs/width_sweep_summary.sh)
echo "  noise summary: $SUM -> $VIZ/noise_generalisation.png"

echo "=== (2) BLENDING ABLATION (feathered tile blend on vs off) on the in-distribution level (noise=1) ==="
if [ -n "${IMJID[1]}" ]; then
    RID=lec_n1; SIMD=$SIMROOT/run$RID; H5=$SIMD/dataset.h5; MS=$SIMD/sim_clean.ms; PREDS=$SIMROOT/preds_$RID.npz
    AW=$(env SIM=1 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS RESET_COL=1 NO_FEATHER=1 \
         sbatch --parsable --dependency=afterok:${IMJID[1]} inference/jobs/inpaint_writeback.sh)
    AI=$(env SIM=1 MS=$MS H5=$H5 INPCOL=INPAINTED_DATA DO_INPAINT=1 DELAY=0 DPSS=0 DPSSFILL=0 MEANFILL=0 \
         OUT=$VIZ/ablation_nofeather \
         sbatch --parsable --dependency=afterok:$AW evaluation/image_eval.sh)
    echo "  no-feather: write $AW -> image $AI  (compare $VIZ/n1 [feathered] vs $VIZ/ablation_nofeather)"
else
    echo "  (noise=1 not in NOISE_SCALES; add it to run the blending ablation)"
fi

echo ""
echo "Already-completed ablations to tabulate alongside these (no rerun needed):"
echo "  - sim-prior finetune vs from-scratch  (finetune std 0.74 vs scratch 0.07; prior essential)"
echo "  - inference noise_floor sweep          (none/0.3/0.5/auto; fakehole_delay_*.npz)"
echo "  - native tiling vs 512-downsample      (real variants v6_native512 vs v5_all512)"
echo ""
echo "watch: squeue -u \$USER"
echo "results: $VIZ/noise_generalisation.png ; $VIZ/n<scale>/image_comparison.png+metrics.json ;"
echo "         $VIZ/ablation_nofeather/metrics.json vs $VIZ/n1/metrics.json (blending on/off)"
