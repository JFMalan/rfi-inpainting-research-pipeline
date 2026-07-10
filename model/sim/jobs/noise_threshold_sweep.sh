#!/bin/bash
# Trainable-noise-threshold sweep (SIMULATED). Fix a normal-brightness sky (sky_model.txt,
# 0.003-0.123 Jy) and vary ONLY the thermal noise: for each NOISE_SCALE, simulate a run,
# train a model ~30ep from scratch, and read the model-vs-mean-fill in-hole MAE that train.py
# logs on its held-out val split. The threshold is the per-baseline SNR where the model stops
# beating mean-fill (no recoverable structure left). Answers "how much noise can we train on".
# Run from repo root:  bash model/sim/jobs/noise_threshold_sweep.sh
#
# sims run in parallel (independent MSes); trainings serialize on the GPU (afterok chain).
set -e
SCALES=${SCALES:-"1.0 0.5 0.25 0.125"}
SKY_MODEL=${SKY_MODEL:-sky_model.txt}   # normal brightness; SAME sky every level so only noise varies
EPOCHS=${EPOCHS:-30}
TRAIN_TIME=${TRAIN_TIME:-24:00:00}      # override train_sim.sh 144h wall; 24h covers 30ep incl V100 + eval
NODELIST=${NODELIST:-}                  # e.g. gpu-007 to pin the training jobs to one node (only if free)
NODEARG=""; [ -n "$NODELIST" ] && NODEARG="--nodelist=$NODELIST"
SIMROOT=/scratch3/users/$USER/rfi/simulated
RUNS=/idia/users/$USER/rfi/runs
VIZ=/idia/users/$USER/rfi/viz/noise_threshold
mkdir -p logs "$VIZ"

echo "=== noise-threshold sweep: sky=$SKY_MODEL  scales=$SCALES  epochs=$EPOCHS ==="
declare -a TRAINJOBS
lowest=""; lowscale=""
prev=""
for S in $SCALES; do
    MILLI=$(python3 -c "print(f'{int(round($S*1000)):04d}')")
    RID=thr_n$MILLI
    H5=$SIMROOT/run$RID/dataset.h5
    SJ=$(env RUN_ID=$RID NOISE_SCALE=$S SKY_MODEL=$SKY_MODEL GEN_RANDOM_SKY=0 \
         sbatch --parsable data_preparation/simulated/jobs/simulate.sh)
    deps="afterok:$SJ"; [ -n "$prev" ] && deps="$deps,afterok:$prev"
    TJ=$(env RUN_ID=$RID EPOCHS=$EPOCHS PHASE=1 \
         sbatch --parsable --time=$TRAIN_TIME $NODEARG --dependency=$deps model/sim/train_sim.sh)
    TRAINJOBS+=("$TJ")
    prev=$TJ
    lowest="$SIMROOT/run$RID/clean_baselines.h5"; lowscale=$S
    echo "  scale=$S  sim $SJ -> train $TJ  (-> $RUNS/phase1_$RID)"
done

# clean_h5 for SNR calibration: use the lowest-noise run (least noise bias on median |V|)
alljobs=$(IFS=,; echo "${TRAINJOBS[*]}")
SUM=$(env RUNS_ROOT=$RUNS SCALES="$SCALES" CLEAN_H5=$lowest OUT=$VIZ/noise_threshold.png \
      sbatch --parsable --dependency=afterok:$alljobs figures/jobs/noise_threshold_summary.sh)
echo "=== summary: $SUM (afterok $alljobs); SNR calibrated from scale=$lowscale clean_baselines.h5 ==="
echo ""
echo "watch: squeue -u \$USER"
echo "result: $VIZ/noise_threshold.png  (model MAE vs mean-fill vs noise; threshold = crossing)"
echo "per-run logs: $RUNS/phase1_thr_n*/log.json  (amp_mae vs amp_mf, beats_mf per epoch)"
