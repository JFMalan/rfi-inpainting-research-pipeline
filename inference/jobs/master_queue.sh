#!/bin/bash
# Master queue: run the write-back oracle gate (sim) and the selective inpaint + imaging (real)
# in parallel. They touch different MSes so they run concurrently; only jobs that edit the SAME MS
# are serialized with afterok (concurrent FLAG edits on one MS would race). Run from repo root:
#   bash inference/jobs/master_queue.sh
set -e

# --- real (selective inpaint) ---
MS=${MS:-/scratch3/users/$USER/rfi/real/inpaint_target.ms}
H5=${H5:-/scratch3/users/$USER/rfi/real/variants/v6_native512.h5}
FT=${FT:-/idia/users/$USER/rfi/runs/phase2_decompose_fullamp/v6_native512_finetune/best.pt}
# --- sim (write-back oracle gate) ---
RUN_ID=${RUN_ID:-1}
SIM_MS=${SIM_MS:-/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/sim_clean.ms}
SIM_H5=${SIM_H5:-/scratch3/users/$USER/rfi/simulated/run${RUN_ID}/dataset.h5}
VIZ=/idia/users/$USER/rfi/viz

echo "=== GROUP A: write-back oracle gate (sim run${RUN_ID}) — validates the write-back mechanics ==="
echo "[A1] level-0 native passthrough (true DATA at holes; ORACLE0 vs clean == write-back mechanics OK?)"
A1=$(env SIM=1 MS=$SIM_MS H5=$SIM_H5 sbatch --parsable inference/jobs/oracle_level0.sh)
echo "  -> $A1"
echo "[A2] phase-fix oracle (cos/sin-resize; isolates the phase-angle-resize suspect). afterok A1 (same MS)"
A2=$(env SIM=1 MS=$SIM_MS H5=$SIM_H5 sbatch --parsable --dependency=afterok:$A1 inference/jobs/oracle_phasefix.sh)
echo "  -> $A2 (waits on $A1)"

echo "=== GROUP B: selective inpaint on real (parallel to A; different MS) ==="
echo "[B-viz] selective spectrogram (finetune/sim/scratch, persistent bands left flagged)"
BV=$(env H5=$H5 FT_CKPT=$FT NF=none KEEP_PERSIST=1 OUT=$VIZ/compare_selective.png MINFF=0.2 MAXFF=0.85 \
     sbatch --parsable model/diagnostics/jobs/compare_models_real.sh)
echo "  -> $BV"
PREDS=/scratch3/users/$USER/rfi/inpaint_preds_selective.npz
echo "[B-infer] GPU inference -> preds (48GB, fits a busy GPU node; SMOOTH=0 for the full-amp finetune)"
BP=$(env SIM=0 CKPT=$FT H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS SMOOTH=0 NOISE_FLOOR=none STEPS=50 \
     sbatch --parsable inference/jobs/inpaint_infer.sh)
echo "  -> $BP"
echo "[B-write] CPU selective write-back on Main (128GB) -> INPAINTED_DATA (non-persistent only). afterok B-infer"
BW=$(env SIM=0 MS=$MS H5=$H5 OUTCOL=INPAINTED_DATA PREDS=$PREDS KEEP_PERSIST=1 UNFLAG=1 \
     sbatch --parsable --dependency=afterok:$BP inference/jobs/inpaint_writeback.sh)
echo "  -> $BW (waits on $BP)"
echo "[B-image] wsclean Flagged-everything vs Selective-inpaint (+DPSS delay). afterok B-write (same MS)"
BI=$(env SIM=0 MS=$MS H5=$H5 KEEP_PERSIST=1 DO_INPAINT=1 DPSS=1 OUT=$VIZ/image_selective \
     sbatch --parsable --dependency=afterok:$BW evaluation/image_eval.sh)
echo "  -> $BI (waits on $BW)"

echo ""
echo "watch: squeue -u \$USER"
echo "READ THE ORACLE FIRST: grep -H 'RMS\\|match\\|ORACLE0\\|done' logs/oracle0-*-stdout.log"
echo "  if ORACLE0 != clean, the selective image below uses the same (buggy) write-back -- treat as suspect."
echo "results: $VIZ/compare_selective.png ; $VIZ/oracle0/image_comparison.png ;"
echo "         $VIZ/oracle_pfix/image_comparison.png ; $VIZ/image_selective/image_comparison.png + delay_spectrum.png"
