#!/bin/bash
#SBATCH --job-name='rfi-width-summary'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/width-summary-%j-stdout.log
#SBATCH --error=logs/width-summary-%j-stderr.log

set -e
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
ROOT=/users/$USER/rfi-inpainting-research-pipeline
SWEEP_ROOT=${SWEEP_ROOT:-/idia/users/$USER/rfi/viz/width_sweep}
WIDTHS=${WIDTHS:-"4 8 16 32 64 128"}
OUT=${OUT:-$SWEEP_ROOT/width_sweep_summary.png}
PREFIX=${PREFIX:-w}                                       # 'w' width sweep, 'n' noise sweep
XLABEL=${XLABEL:-RFI band width (native channels)}
LOGX=${LOGX:-1}                                           # 1 = log2 x (width); 0 = linear (noise)
LOGX_ARG=""; [ "$LOGX" = "1" ] && LOGX_ARG="--logx"

singularity exec $ASTROPY python $ROOT/evaluation/plot_width_sweep.py \
    --root "$SWEEP_ROOT" --widths $WIDTHS --out "$OUT" --prefix "$PREFIX" --xlabel "$XLABEL" $LOGX_ARG
echo "done -> $OUT"
