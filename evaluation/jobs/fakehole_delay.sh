#!/bin/bash
#SBATCH --job-name='rfi-fakehole-delay'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/fakehole-delay-%j-stdout.log
#SBATCH --error=logs/fakehole-delay-%j-stderr.log

set -e

# Honest REAL headline: fake holes over GOOD real data (known truth), fill with the model vs
# DPSS vs zero, compare delay spectra to the true good data. The RFI-gap delay eval can't do
# this (real flags have no RFI-free truth), so this is the only ground-truthed real delay test.
H5=${H5:?set H5=/path/to/v6_native512.h5}
CKPT=${CKPT:?set CKPT=/path/to/real/best.pt}
OUT=${OUT:-/idia/users/$USER/rfi/viz/fakehole_delay.npz}
STEPS=${STEPS:-50}
MAX_UNITS=${MAX_UNITS:-300}
DPSS_HW=${DPSS_HW:-0.1}
NOISE_FLOORS=${NOISE_FLOORS:-none}   # sweep e.g. NOISE_FLOORS="none 0.3 0.5 auto"
GPR_ELL=${GPR_ELL:-30}               # GPR kernel length scale (channels); sweep for a fair classical bar
HOLE_MODE=${HOLE_MODE:-mixed}        # mixed | blob | band (band = persistent-band-shaped ceiling test)
FRAC_RANGE=${FRAC_RANGE:-}           # "lo hi" fake-hole fraction range (eval default 0.1 0.25)
MAX_FLAG=${MAX_FLAG:-0.85}           # only score tiles below this real-flag fraction; heavily-flagged
                                     # tiles have little RFI-free structure so the metric saturates
POST_SAMPLE=${POST_SAMPLE:-0}        # 1 = also score a genuine posterior-sample ensemble (eta>0/repaint)
ETA=${ETA:-1.0}
REPAINT_U=${REPAINT_U:-1}
ENSEMBLE=${ENSEMBLE:-4}
NO_DIVNORM=${NO_DIVNORM:-0}          # 1 = score in normalised space (divisor test)

EXTRA=""
[ "$POST_SAMPLE" = "1" ] && EXTRA="$EXTRA --post-sample --eta $ETA --repaint-u $REPAINT_U --ensemble $ENSEMBLE"
[ "$NO_DIVNORM" = "1" ] && EXTRA="$EXTRA --no-divnorm"
[ -n "$FRAC_RANGE" ] && EXTRA="$EXTRA --frac-range $FRAC_RANGE"

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
mkdir -p logs $(dirname $OUT)

LIBDIR=/usr/lib/x86_64-linux-gnu
LIBCUDA=$(ls $LIBDIR/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls $LIBDIR/libnvidia-ml.so.*.* 2>/dev/null | head -1)
if [ -z "$LIBCUDA" ] || [ -z "$LIBNVML" ]; then echo "no driver libs on $(hostname)"; exit 1; fi
NVBIND="--bind $LIBCUDA:$LIBDIR/libcuda.so.1 --bind $LIBNVML:$LIBDIR/libnvidia-ml.so.1"

singularity exec --nv $NVBIND $GPU python $ROOT/evaluation/fakehole_delay_eval.py \
    --h5 "$H5" --ckpt "$CKPT" --out "$OUT" \
    --steps $STEPS --max-units $MAX_UNITS --max-flag-frac $MAX_FLAG --dpss-hw $DPSS_HW \
    --noise-floors $NOISE_FLOORS --gpr-ell $GPR_ELL --hole-mode $HOLE_MODE $EXTRA

echo "done -> $OUT"
