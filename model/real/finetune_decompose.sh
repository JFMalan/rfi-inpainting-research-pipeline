#!/bin/bash
#SBATCH --job-name='rfi-finetune-decompose'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A40
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=18:00:00
#SBATCH --output=logs/finetune-decompose-%j-stdout.log
#SBATCH --error=logs/finetune-decompose-%j-stderr.log

set -e

# Decompose-then-inpaint on REAL data: fine-tune the sim decompose model on real with
# --smooth-target (self-sup target = recoverable smooth amplitude), eval vs interp +
# mean-fill ON THE SMOOTH TARGET. Also runs from-scratch for the sim-prior comparison.
VARIANTS=${VARIANTS:-"v6_native512"}
INIT=${INIT:-/idia/users/$USER/rfi/runs/phase1_all_decompose_tiled80ep/best.pt}
ITERS=${ITERS:-8000}
BATCH=${BATCH:-4}
LR=${LR:-4e-5}         # fine-tune LR (5x below from-scratch 2e-4) so the sim prior isn't blown away
EMA=${EMA:-0.999}      # fast EMA: 0.9999 froze the shadow at the init over a short run (audit)
SIGMA=${SIGMA:-1.0}    # sigma sweep on real: cleanly splits structure (smooth ac~0.92) from white noise (grain ac~0.01)
SMOOTH=${SMOOTH:-1}    # 1 = smooth-target objective (decompose); 0 = full-amplitude objective
FAKE_MASK_MODE=${FAKE_MASK_MODE:-mixed}   # mixed = train on the real RFI geometry (freq bands + bursts + blobs); '2d' = blobs only (legacy)
DO_SCRATCH=${DO_SCRATCH:-1}
if [ "$SMOOTH" = "1" ]; then SMOOTH_ARG="--smooth-target --smooth-sigma $SIGMA"; else SMOOTH_ARG=""; echo "FULL-AMPLITUDE objective (no smooth-target)"; fi

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
VARDIR=/scratch3/users/$USER/rfi/real/variants
RUNROOT=/idia/users/$USER/rfi/runs/phase2_decompose${TAG:+_$TAG}

mkdir -p $RUNROOT logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

if [ ! -f "$INIT" ]; then
    echo "sim decompose checkpoint not found: $INIT (is train_sim_decompose done?)"; exit 1
fi

run_one () {
    local NAME=$1 MODE=$2 H5=$3 OUT=$4 RLR=$5 EXTRA=$6
    echo ""
    echo "======== TRAIN $NAME [$MODE] (decompose, lr=$RLR ema=$EMA) ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/train_real.py \
        --data $H5 --out $OUT --epochs 1000 --batch-size $BATCH --max-iters $ITERS --lr $RLR \
        --ema-decay $EMA --sample-every 4 --val-eval-patches 24 --min-epochs 8 --min-delta 0.005 --patience 6 \
        --fake-mask-mode $FAKE_MASK_MODE $SMOOTH_ARG $EXTRA || { echo "train failed $NAME $MODE"; return; }
    echo "======== EVAL  $NAME [$MODE] ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/real/eval_real.py \
        --data $H5 --ckpt $OUT/best.pt --tag ${NAME}_${MODE}${TAG:+_$TAG} --batch-size $BATCH \
        $SMOOTH_ARG || echo "eval failed $NAME $MODE"
}

for V in $VARIANTS; do
    H5=$VARDIR/${V}.h5
    run_one $V finetune $H5 $RUNROOT/${V}_finetune $LR "--init-from $INIT"
    if [ "$DO_SCRATCH" = "1" ]; then
        run_one $V scratch $H5 $RUNROOT/${V}_scratch 2e-4 ""
    fi
done

echo ""
echo "============ PHASE-2 DECOMPOSE RANKING (fakeMAE vs smooth target; lower better) ============"
echo -e "run\tTRE\tTRE_mf\tfakeMAE\tinterp\tmean-fill\tnfr\tn"
grep -h "^RESULTLINE" logs/finetune-decompose-${SLURM_JOB_ID}-stdout.log | cut -f2-
