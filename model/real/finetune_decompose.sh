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
VARIANTS=${VARIANTS:-"v1_upsample512"}
INIT=${INIT:-/idia/users/$USER/rfi/runs/phase1_all_decompose/best.pt}
ITERS=${ITERS:-8000}
BATCH=${BATCH:-4}
LR=${LR:-2e-4}
SIGMA=${SIGMA:-1.0}     # sigma sweep on real: cleanly splits structure (smooth ac~0.92) from white noise (grain ac~0.01)
DO_SCRATCH=${DO_SCRATCH:-1}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
VARDIR=/scratch3/users/$USER/rfi/real/variants
RUNROOT=/idia/users/$USER/rfi/runs/phase2_decompose

mkdir -p $RUNROOT logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

if [ ! -f "$INIT" ]; then
    echo "sim decompose checkpoint not found: $INIT (is train_sim_decompose done?)"; exit 1
fi

run_one () {
    local NAME=$1 MODE=$2 H5=$3 OUT=$4 EXTRA=$5
    echo ""
    echo "======== TRAIN $NAME [$MODE] (decompose) ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/train_real.py \
        --data $H5 --out $OUT --epochs 1000 --batch-size $BATCH --max-iters $ITERS --lr $LR \
        --sample-every 4 --val-eval-patches 24 --min-epochs 4 --min-delta 0.02 --patience 4 \
        --smooth-target --smooth-sigma $SIGMA $EXTRA || { echo "train failed $NAME $MODE"; return; }
    echo "======== EVAL  $NAME [$MODE] (decompose, vs smooth target) ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/real/eval_real.py \
        --data $H5 --ckpt $OUT/best.pt --tag ${NAME}_${MODE}_decompose --batch-size $BATCH \
        --smooth-target --smooth-sigma $SIGMA || echo "eval failed $NAME $MODE"
}

for V in $VARIANTS; do
    H5=$VARDIR/${V}.h5
    run_one $V finetune $H5 $RUNROOT/${V}_finetune "--init-from $INIT"
    if [ "$DO_SCRATCH" = "1" ]; then
        run_one $V scratch $H5 $RUNROOT/${V}_scratch ""
    fi
done

echo ""
echo "============ PHASE-2 DECOMPOSE RANKING (fakeMAE vs smooth target; lower better) ============"
echo -e "run\tTRE\tTRE_mf\tfakeMAE\tinterp\tmean-fill\tn"
grep -h "^RESULTLINE" logs/finetune-decompose-${SLURM_JOB_ID}-stdout.log | cut -f2-
