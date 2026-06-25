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

# Two-stage: GPU container has torch but NOT casacore/skimage; ASTRO-PY3.10 has casacore+
# skimage but no torch. So infer on GPU (save preds), then write the MS on CPU.
# UNVALIDATED on real data — run the sim round-trip (docs/ms-writeback-plan.md) first.
# SIM=1 -> sim dataset.h5 (corrupted/mask); SIM=0 -> real (data/flags).
SIM=${SIM:-0}
TAG=${TAG:-phase1_all_decompose_80ep}
SMOOTH=${SMOOTH:-1}
# none = low-distortion conditional-mean fill, the right mode for IMAGING (default).
# auto = adds resampled noise for a statistically-consistent single-baseline waterfall,
# but that injected noise only HURTS image fidelity — use it for the waterfall viz, not imaging.
NOISE_FLOOR=${NOISE_FLOOR:-none}
STEPS=${STEPS:-200}
BATCH=${BATCH:-8}
OUTCOL=${OUTCOL:-INPAINTED_DATA}
UNFLAG=${UNFLAG:-0}
MAX_UNITS=${MAX_UNITS:-}
# WEIGHT_FRAC: down-weight inpainted pixels in WEIGHT_SPECTRUM to this fraction of the real
# weight instead of hard-substituting at full weight. 0~=flagged, 1=full. Unset = old behaviour.
WEIGHT_FRAC=${WEIGHT_FRAC:-}

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
CKPT=${CKPT:-/idia/users/$USER/rfi/runs/${TAG}/best.pt}
MS=${MS:?set MS=/path/to/flagged.ms}
H5=${H5:?set H5=/path/to/extracted_dataset.h5}
PREDS=${PREDS:-/scratch3/users/$USER/rfi/inpaint_preds_${SLURM_JOB_ID}.npz}

if [ "${ORACLE:-0}" != "1" ] && [ ! -f "$CKPT" ]; then echo "checkpoint not found: $CKPT"; exit 1; fi

mkdir -p logs
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then
    echo "could not find driver libs on $(hostname) in $LIBDIR"; exit 1
fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

INFER_EXTRA=""; WRITE_EXTRA=""
[ "$SIM" = "1" ] && { INFER_EXTRA="$INFER_EXTRA --sim"; WRITE_EXTRA="$WRITE_EXTRA --sim"; }
[ "$SMOOTH" = "1" ] && INFER_EXTRA="$INFER_EXTRA --smooth-target"
[ "$ORACLE" = "1" ] && INFER_EXTRA="$INFER_EXTRA --oracle"
[ "$UNFLAG" = "1" ] && WRITE_EXTRA="$WRITE_EXTRA --unflag"
[ -n "$WEIGHT_FRAC" ] && WRITE_EXTRA="$WRITE_EXTRA --weight-frac $WEIGHT_FRAC"
[ -n "$MAX_UNITS" ] && INFER_EXTRA="$INFER_EXTRA --max-units $MAX_UNITS"

echo "STAGE 1 (GPU infer) node $(hostname)  ckpt=$CKPT  h5=$H5 -> $PREDS"
singularity exec --nv $NVBIND $GPU python $ROOT/inference/inpaint_infer.py \
    --h5 "$H5" --ckpt "$CKPT" --out-preds "$PREDS" \
    --steps $STEPS --batch $BATCH --noise-floor $NOISE_FLOOR $INFER_EXTRA

echo "STAGE 2 (CPU write-back, ASTRO-PY3.10)  $PREDS -> $OUTCOL in $MS"
singularity exec $ASTROPY python $ROOT/inference/inpaint_write.py \
    --ms "$MS" --h5 "$H5" --preds "$PREDS" --out-col "$OUTCOL" $WRITE_EXTRA

echo "done -> $OUTCOL in $MS"
