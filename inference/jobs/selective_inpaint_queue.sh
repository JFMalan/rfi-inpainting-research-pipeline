#!/bin/bash
# Selective inpaint: fill only the NON-persistent RFI (leave the wide persistent bands flagged),
# then image with wsclean to test whether recovering that ~60% of RFI-hit bandwidth improves the
# continuum map vs flagging everything. Run from the repo root on ilifu:
#   bash inference/jobs/selective_inpaint_queue.sh
#
# MS and H5 MUST be the same observation (the write-back maps h5 units -> MS rows). H5 is the
# tiled v6 extraction; MS is your writable copy of that observation.
set -e
MS=${MS:-/scratch3/users/$USER/rfi/real/inpaint_target.ms}
H5=${H5:-/scratch3/users/$USER/rfi/real/variants/v6_native512.h5}
FT=${FT:-/idia/users/$USER/rfi/runs/phase2_decompose_fullamp/v6_native512_finetune/best.pt}
VIZ=/idia/users/$USER/rfi/viz

echo "[viz] selective-inpaint spectrogram (finetune/sim/scratch, persistent bands left flagged)"
V=$(env H5=$H5 FT_CKPT=$FT NF=none KEEP_PERSIST=1 OUT=$VIZ/compare_selective.png MINFF=0.2 MAXFF=0.85 \
    sbatch --parsable model/diagnostics/jobs/compare_models_real.sh)
echo "  -> viz $V"

PREDS=/scratch3/users/$USER/rfi/inpaint_preds_selective.npz
echo "[infer] GPU inference -> preds (48GB, SMOOTH=0 full-amp finetune, noise_floor=none for imaging)"
P=$(env SIM=0 CKPT=$FT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS SMOOTH=0 NOISE_FLOOR=none STEPS=50 \
    sbatch --parsable inference/jobs/inpaint_infer.sh)
echo "  -> infer $P"

echo "[write] CPU selective write-back on Main (128GB) -> INPAINTED_DATA (non-persistent only)"
W=$(env SIM=0 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS KEEP_PERSIST=1 UNFLAG=1 \
    sbatch --parsable --dependency=afterok:$P inference/jobs/inpaint_writeback.sh)
echo "  -> write-back $W (waits on $P)"

echo "[image] wsclean: Flagged-everything vs Selective-inpaint (+DPSS delay baseline), after the write-back"
I=$(env SIM=0 MS=$MS H5=$H5 KEEP_PERSIST=1 DO_INPAINT=1 DPSS=1 OUT=$VIZ/image_selective \
    sbatch --parsable --dependency=afterok:$W evaluation/image_eval.sh)
echo "  -> image $I (waits on $W)"

echo ""
echo "watch: squeue -u \$USER"
echo "results: $VIZ/compare_selective.png ; $VIZ/image_selective/image_comparison.png ; $VIZ/image_selective/delay_spectrum.png"
