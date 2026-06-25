#!/bin/bash
#SBATCH --job-name='rfi-image-eval'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/image-eval-%j-stdout.log
#SBATCH --error=logs/image-eval-%j-stderr.log

set -e

# Sky-map comparison of the write-back: Clean (DATA) vs Flagged (DATA, holes flagged)
# vs Inpainted (INPAINTED_DATA). Needs a FULL write-back first (all baselines) — imaging
# a 50-unit subset is meaningless. wsclean defaults follow oxkat (briggs -0.3, auto-mask 4).
# NOTE: temporarily sets FLAG at the holes for the flagged image, then clears it back.
# SIM=1 also produces the Clean truth image; on real data there's no Clean reference.
SIM=${SIM:-1}
MS=${MS:?set MS=/path/to/ms}
H5=${H5:?set H5=/path/to/dataset.h5}
INPCOL=${INPCOL:-INPAINTED_DATA}
IMSIZE=${IMSIZE:-2048}
CELL=${CELL:-2asec}
NITER=${NITER:-10000}
MAX_UNITS=${MAX_UNITS:-}
DO_INPAINT=${DO_INPAINT:-1}   # 0 = skip the inpainted column (e.g. flagged+mean-fill preview before a model exists)
MEANFILL=${MEANFILL:-0}       # 1 = also write + image a per-channel time-mean fill (3-way benchmark)

ROOT=/users/$USER/rfi-inpainting-research-pipeline
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
OXKAT_SIF=$(ls /idia/software/containers/ | grep -i oxkat | head -1)
OXKAT=/idia/software/containers/$OXKAT_SIF
OUT=${OUT:-/idia/users/$USER/rfi/viz/image_eval}
IMG=$OUT/img
mkdir -p $IMG logs

SIMARG=""; [ "$SIM" = "1" ] && SIMARG="--sim"
MU=""; [ -n "$MAX_UNITS" ] && MU="--max-units $MAX_UNITS"
echo "oxkat=$OXKAT_SIF  ms=$MS  imsize=$IMSIZE cell=$CELL"

wsc () {  # name  column
    singularity exec $OXKAT wsclean -name $IMG/$1 -data-column $2 \
        -size $IMSIZE $IMSIZE -scale $CELL -niter $NITER -mgain 0.9 \
        -weight briggs -0.3 -auto-mask 4 -auto-threshold 1 -use-wgridder -channels-out 1 \
        -no-update-model-required "$MS"
}

set_flag () { singularity exec $ASTROPY python $ROOT/evaluation/set_holes_flag.py \
    --ms "$MS" --h5 "$H5" --mode $1 $SIMARG $MU; }

# ensure holes are unflagged, then image truth + the filled columns (holes cleared -> wsclean uses the fill)
set_flag clear
if [ "$DO_INPAINT" = "1" ]; then echo "==== image Inpainted ($INPCOL) ===="; wsc inpainted $INPCOL; fi
if [ "$MEANFILL" = "1" ]; then
    echo "==== mean-fill write + image (per-channel time-mean) ===="
    singularity exec $ASTROPY python $ROOT/evaluation/mean_fill_write.py --ms "$MS" --h5 "$H5" --out-col MEANFILL_DATA $SIMARG
    wsc meanfill MEANFILL_DATA
fi
if [ "$SIM" = "1" ]; then echo "==== image Clean (DATA truth) ===="; wsc clean DATA; fi

# flag the holes, image the flag-only (missing-data) case, then restore
echo "==== flag holes -> image Flagged (DATA) ===="
set_flag set
wsc flagged DATA
set_flag clear

CLEAN_ARG=""; [ "$SIM" = "1" ] && CLEAN_ARG="--clean $IMG/clean-image.fits"
INP_ARG=""; [ "$DO_INPAINT" = "1" ] && INP_ARG="--inpainted $IMG/inpainted-image.fits"
MEANFILL_ARG=""; [ "$MEANFILL" = "1" ] && MEANFILL_ARG="--meanfill $IMG/meanfill-image.fits"
echo "==== compare (continuum image) ===="
singularity exec $ASTROPY python $ROOT/evaluation/compare_images.py \
    $CLEAN_ARG --flagged $IMG/flagged-image.fits $MEANFILL_ARG $INP_ARG \
    --out $OUT/image_comparison.png

if [ "$DO_INPAINT" = "1" ]; then
    echo "==== compare (delay space) ===="
    singularity exec $ASTROPY python $ROOT/evaluation/delay_spectrum.py \
        --ms "$MS" --h5 "$H5" --inp-col $INPCOL --out $OUT/delay_spectrum.png $SIMARG $MU
fi

echo "done -> $OUT/image_comparison.png  $OUT/delay_spectrum.png  (fits in $IMG/)"
