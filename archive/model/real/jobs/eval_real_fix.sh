#!/bin/bash
#SBATCH --job-name='rfi-eval-real-fix'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-real-fix-%j-stdout.log
#SBATCH --error=logs/eval-real-fix-%j-stderr.log

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model/real
DATA=${DATA:-/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5}
RUNS=/idia/users/$USER/rfi/runs
MAX_EVAL=${MAX_EVAL:-64}
mkdir -p logs

LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
echo "node $(hostname)  data $DATA"

# tag | ckpt | smooth-flag  (decompose ckpts MUST eval with --smooth-target for obs/target parity)
# phase1_all = best sim-trained model (full-amp) applied to real -> raw context, no smooth flag
CONFIGS=(
  "sim_phase1_all|$RUNS/phase1_all/best.pt|"
  "decompose_finetune|$RUNS/phase2_decompose/v1_upsample512_finetune/best.pt|--smooth-target"
  "decompose_scratch|$RUNS/phase2_decompose/v1_upsample512_scratch/best.pt|--smooth-target"
  "fullamp_finetune|$RUNS/phase2_decompose_fullamp/v1_upsample512_finetune/best.pt|"
  "fullamp_scratch|$RUNS/phase2_decompose_fullamp/v1_upsample512_scratch/best.pt|"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r TAG CKPT SMOOTH <<< "$entry"
  if [ ! -f "$CKPT" ]; then echo "SKIP $TAG (missing $CKPT)"; continue; fi
  echo "==== $TAG ===="
  singularity exec --nv $NVBIND $GPU python $SCRIPTS/eval_real.py \
      --data $DATA --ckpt $CKPT --tag $TAG --max-eval $MAX_EVAL $SMOOTH
done

echo "done  (grep RESULTLINE in this log: tag, TRE, mean-fill TRE, fake-MAE, interp, mean-fill MAE, noise-floor-ratio, n)"
