#!/bin/bash
#SBATCH --job-name='rfi-finetune'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=18:00:00
#SBATCH --output=logs/finetune-%j-stdout.log
#SBATCH --error=logs/finetune-%j-stderr.log

set -e

# Phase-2 decisive test, fair head-to-head: for each variant, train BOTH
# fine-tuned-from-sim AND from-scratch at the SAME 8000-iter budget, eval each on
# the held-out test vs interp + mean-fill. Answers: does the sim prior help, and
# does more real data (v4) beat clean-but-tiny (v1)?
VARIANTS=${VARIANTS:-"v1_upsample512 v4_relaxed512"}
INIT=${INIT:-/idia/users/$USER/rfi/runs/phase1_all/best.pt}
ITERS=${ITERS:-8000}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}
DO_SCRATCH=${DO_SCRATCH:-1}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
VARDIR=/scratch3/users/$USER/rfi/real/variants
RUNROOT=/idia/users/$USER/rfi/runs/phase2

mkdir -p $RUNROOT logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

if [ ! -f "$INIT" ]; then
    echo "sim checkpoint not found: $INIT (is the sim training done?)"; exit 1
fi

run_one () {
    local NAME=$1 MODE=$2 H5=$3 OUT=$4 EXTRA=$5
    echo ""
    echo "======== TRAIN $NAME [$MODE] ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/train_real.py \
        --data $H5 --out $OUT --epochs 1000 --batch-size $BATCH --max-iters $ITERS --lr $LR \
        --sample-every 4 --val-eval-patches 24 --min-epochs 4 --min-delta 0.02 --patience 4 \
        $EXTRA || { echo "train failed $NAME $MODE"; return; }
    echo "======== EVAL  $NAME [$MODE] ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/real/eval_real.py \
        --data $H5 --ckpt $OUT/best.pt --tag ${NAME}_${MODE} --batch-size $BATCH || echo "eval failed $NAME $MODE"
}

for V in $VARIANTS; do
    H5=$VARDIR/${V}.h5
    run_one $V finetune $H5 $RUNROOT/${V}_finetune "--init-from $INIT"
    if [ "$DO_SCRATCH" = "1" ]; then
        run_one $V scratch $H5 $RUNROOT/${V}_scratch ""
    fi
done

echo ""
echo "============ PHASE-2 RANKING (fakeMAE: lower better; vs interp + mean-fill) ============"
echo -e "run\tTRE\tTRE_mf\tfakeMAE\tinterp\tmean-fill\tn"
grep -h "^RESULTLINE" logs/finetune-${SLURM_JOB_ID}-stdout.log | cut -f2-
