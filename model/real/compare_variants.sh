#!/bin/bash
#SBATCH --job-name='rfi-compare-variants'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/compare-variants-%j-stdout.log
#SBATCH --error=logs/compare-variants-%j-stderr.log

set -e

ITERS=${ITERS:-1500}
BATCH=${BATCH:-4}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
VARDIR=/scratch3/users/$USER/rfi/real/variants
RUNROOT=/idia/users/$USER/rfi/runs/variant_compare

mkdir -p $RUNROOT logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

# train each variant to an equal ITER budget (fair across variants of different size),
# then score the held-out TEST baselines.
for H5 in $VARDIR/*.h5; do
    NAME=$(basename $H5 .h5)
    OUT=$RUNROOT/$NAME
    mkdir -p $OUT
    echo ""
    echo "======== TRAIN $NAME (max-iters $ITERS) ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/train_real.py \
        --data $H5 --out $OUT --epochs 1000 --batch-size $BATCH --max-iters $ITERS || {
            echo "train failed for $NAME"; continue; }

    echo "======== EVAL  $NAME (held-out test) ========"
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/real/eval_real.py \
        --data $H5 --ckpt $OUT/best.pt --tag $NAME --batch-size $BATCH || {
            echo "eval failed for $NAME"; continue; }
done

echo ""
echo "============ VARIANT RANKING (lower TRE / fake-MAE = better) ============"
echo -e "variant\tTRE\tTRE_mf\tfakeMAE\tfakeMAE_mf\tn"
grep -h "^RESULTLINE" logs/compare-variants-${SLURM_JOB_ID}-stdout.log | cut -f2- | sort -k2 -n
