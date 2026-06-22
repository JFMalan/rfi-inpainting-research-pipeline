#!/bin/bash
#SBATCH --job-name='rfi-beat-meanfill'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/beat-meanfill-%j-stdout.log
#SBATCH --error=logs/beat-meanfill-%j-stderr.log

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

# complex-vis uses ch0 amplitude: full-amp/sim ckpts -> true-scale V (the deliverable);
# decompose ckpt -> smooth-amp V (reference only). phase channel is identical across all.
CONFIGS=(
  "sim_phase1_all|$RUNS/phase1_all/best.pt|"
  "fullamp_finetune|$RUNS/phase2_decompose_fullamp/v1_upsample512_finetune/best.pt|"
  "fullamp_scratch|$RUNS/phase2_decompose_fullamp/v1_upsample512_scratch/best.pt|"
  "decompose_finetune|$RUNS/phase2_decompose/v1_upsample512_finetune/best.pt|--smooth-target"
)

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r TAG CKPT SMOOTH <<< "$entry"
  if [ ! -f "$CKPT" ]; then echo "SKIP $TAG (missing $CKPT)"; continue; fi
  echo "==== $TAG ===="
  singularity exec --nv $NVBIND $GPU python $SCRIPTS/beat_meanfill.py \
      --data $DATA --ckpt $CKPT --tag $TAG --max-eval $MAX_EVAL $SMOOTH
done

echo "done  (RESULTLINE: tag, cvis-model, cvis-meanfill, phase-model, phase-meanfill, n)"
