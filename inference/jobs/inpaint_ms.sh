#!/bin/bash
#SBATCH --job-name='rfi-inpaint-ms'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/inpaint-ms-%j-stdout.log
#SBATCH --error=logs/inpaint-ms-%j-stderr.log

set -e

# UNVALIDATED: run the sim round-trip (docs/ms-writeback-plan.md) before pointing at real data.
# SIM=1 -> sim dataset.h5 (keys corrupted/mask); SIM=0 -> real (data/flags).
SIM=${SIM:-0}
TAG=${TAG:-phase1_all_decompose_80ep}
SMOOTH=${SMOOTH:-1}
NOISE_FLOOR=${NOISE_FLOOR:-auto}
STEPS=${STEPS:-200}
OUTCOL=${OUTCOL:-INPAINTED_DATA}
UNFLAG=${UNFLAG:-0}
MAX_UNITS=${MAX_UNITS:-}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/${TAG}/best.pt}
MS=${MS:?set MS=/path/to/flagged.ms}
H5=${H5:?set H5=/path/to/extracted_dataset.h5}

if [ ! -f "$CKPT" ]; then echo "checkpoint not found: $CKPT"; exit 1; fi

mkdir -p logs
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

# fail fast if the container lacks any of torch(GPU)/casacore/skimage rather than guessing.
echo "verifying container has torch+cuda, casacore, skimage"
singularity exec --nv $NVBIND $GPU python -c "import torch; assert torch.cuda.is_available(); import casacore.tables; import skimage; print('deps OK', torch.cuda.get_device_name(0))" || {
    echo "MISSING DEPS in $GPU. Use the two-stage path (GPU infer -> save preds; ASTRO-PY3.10 write-back). See docs/ms-writeback-plan.md."; exit 2; }

EXTRA=""
[ "$SIM" = "1" ] && EXTRA="$EXTRA --sim"
[ "$SMOOTH" = "1" ] && EXTRA="$EXTRA --smooth-target"
[ "$UNFLAG" = "1" ] && EXTRA="$EXTRA --unflag"
[ -n "$MAX_UNITS" ] && EXTRA="$EXTRA --max-units $MAX_UNITS"

echo "node $(hostname)  ms=$MS  h5=$H5  ckpt=$CKPT  out=$OUTCOL  sim=$SIM"
singularity exec --nv $NVBIND $GPU python $ROOT/inference/inpaint_ms.py \
    --ms "$MS" --h5 "$H5" --ckpt "$CKPT" --out-col "$OUTCOL" \
    --steps $STEPS --noise-floor $NOISE_FLOOR $EXTRA

echo "done -> $OUTCOL in $MS"
