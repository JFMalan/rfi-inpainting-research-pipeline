#!/bin/bash
#SBATCH --job-name='rfi-inpaint-infer'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/inpaint-infer-%j-stdout.log
#SBATCH --error=logs/inpaint-infer-%j-stderr.log

set -e

# Stage 1 only: GPU inference -> preds .npz. Split out from archive/inference/jobs/inpaint_ms.sh so the GPU job needs
# modest memory (fits a busy GPU node); the heavy 128GB write-back runs separately on Main via
# inpaint_writeback.sh, reading the same PREDS. Pass a fixed PREDS shared by both jobs.
SIM=${SIM:-0}
TAG=${TAG:-phase1_all_decompose_80ep}
SMOOTH=${SMOOTH:-1}   # 1 = smooth-target (decompose model); 0 = full-amplitude model (e.g. the fullamp finetune)
NOISE_FLOOR=${NOISE_FLOOR:-none}   # none = conditional-mean fill (right for imaging)
STEPS=${STEPS:-200}
BATCH=${BATCH:-8}
OUTCOL=${OUTCOL:-INPAINTED_DATA}
MAX_UNITS=${MAX_UNITS:-}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/${TAG}/best.pt}
H5=${H5:?set H5=/path/to/extracted_dataset.h5}
PREDS=${PREDS:-/scratch3/users/$USER/rfi/inpaint_preds_${OUTCOL}.npz}

if [ "${ORACLE:-0}" != "1" ] && [ ! -f "$CKPT" ]; then echo "checkpoint not found: $CKPT"; exit 1; fi
if [ "$SIM" = "0" ] && [ "${ORACLE:-0}" != "1" ]; then
    case "$CKPT" in
        *phase1_*) echo "ERROR: SIM=0 (real) but CKPT is a sim model ($CKPT)."; exit 1;;
    esac
fi

mkdir -p logs $(dirname "$PREDS")
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then echo "no driver libs on $(hostname)"; exit 1; fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

INFER_EXTRA=""
[ "$SIM" = "1" ] && INFER_EXTRA="$INFER_EXTRA --sim"
[ "$SMOOTH" = "1" ] && INFER_EXTRA="$INFER_EXTRA --smooth-target"
[ "$ORACLE" = "1" ] && INFER_EXTRA="$INFER_EXTRA --oracle"
[ -n "$MAX_UNITS" ] && INFER_EXTRA="$INFER_EXTRA --max-units $MAX_UNITS"

echo "GPU infer node $(hostname)  ckpt=$CKPT  h5=$H5 -> $PREDS"
singularity exec --nv $NVBIND $GPU python $ROOT/inference/inpaint_infer.py \
    --h5 "$H5" --ckpt "$CKPT" --out-preds "$PREDS" \
    --steps $STEPS --batch $BATCH --noise-floor $NOISE_FLOOR $INFER_EXTRA

echo "done -> $PREDS"
