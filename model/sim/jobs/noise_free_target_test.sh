#!/bin/bash
# Noise-free-target test. Train at full thermal noise (input = noisy+RFI) but with the target =
# the noise-free clean signal, then check at write-back whether resampled grain in the filled hole
# matches the surrounding noise (visually + in delay space). Tests two things the pipeline has
# never actually run: (1) does a clean target avoid the rigged interp-in-hole metric, (2) does
# smooth-fill + matched grain stay seamless (no edge discontinuity / delay leakage).
# Run from repo root:  bash model/sim/jobs/noise_free_target_test.sh
#
# Needs the noisy run to already exist (runthr_n<NOISY_MILLI> from the noise-threshold sweep).
# Simulates a NOISE_SCALE=0 twin (same sky/seed), pairs them, trains ~30ep, then fill-checks.
set -e
NOISE_SCALE=${NOISE_SCALE:-1.0}                 # noise level of the input/context run (must exist)
SKY_MODEL=${SKY_MODEL:-sky_model.txt}           # same sky as the noisy run
EPOCHS=${EPOCHS:-30}
TRAIN_TIME=${TRAIN_TIME:-12:00:00}
NODELIST=${NODELIST:-}
NODEARG=""; [ -n "$NODELIST" ] && NODEARG="--nodelist=$NODELIST"
SIMROOT=/scratch3/users/$USER/rfi/simulated
RUNS=/idia/users/$USER/rfi/runs
VIZ=/idia/users/$USER/rfi/viz/noise_threshold
mkdir -p logs "$VIZ"

MILLI=$(python3 -c "print(f'{int(round($NOISE_SCALE*1000)):04d}')")
NOISY=$SIMROOT/runthr_n$MILLI/dataset.h5
CLEAN=$SIMROOT/runthr_n0000/dataset.h5
PAIRED=$SIMROOT/runthr_paired/dataset.h5
[ -f "$NOISY" ] || { echo "noisy run missing: $NOISY (run the noise-threshold sweep first)"; exit 1; }

echo "=== noise-free-target test: input=$MILLI x SEFD, target=noise-free, epochs=$EPOCHS ==="
if [ -f "$CLEAN" ]; then
    echo "reusing existing noise-free twin $CLEAN"; SJ=""
else
    SJ=$(env RUN_ID=thr_n0000 NOISE_SCALE=0 SKY_MODEL=$SKY_MODEL GEN_RANDOM_SKY=0 \
         sbatch --parsable data_preparation/simulated/jobs/simulate.sh)
    echo "  sim noise-free twin: $SJ"
fi

dep=""; [ -n "$SJ" ] && dep="--dependency=afterok:$SJ"
PJ=$(env NOISY=$NOISY CLEAN=$CLEAN OUT=$PAIRED \
     sbatch --parsable $dep evaluation/jobs/pair_dataset.sh)
echo "  pair dataset: $PJ -> $PAIRED"

TJ=$(env RUN_ID=thr_paired EPOCHS=$EPOCHS PHASE=1 \
     sbatch --parsable --time=$TRAIN_TIME $NODEARG --dependency=afterok:$PJ model/sim/train_sim.sh)
echo "  train (clean target): $TJ -> $RUNS/phase1_thr_paired"

CJ=$(env H5=$PAIRED CKPT=$RUNS/phase1_thr_paired/best.pt OUT=$VIZ/fill_check.png N=4 \
     sbatch --parsable --dependency=afterok:$TJ model/diagnostics/jobs/noise_free_fill_check.sh)
echo "  fill-check: $CJ -> $VIZ/fill_check.png"
echo ""
echo "watch: squeue -u \$USER"
echo "result: $VIZ/fill_check.png  (observed | noise-free target | smooth fill | +matched grain | delay spectrum)"
echo "  grain/obs hi-delay ratio ~1 = seamless; smooth/obs <1 = the discontinuity you were worried about"
