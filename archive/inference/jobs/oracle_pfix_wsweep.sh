#!/bin/bash
#SBATCH --job-name='rfi-pfix-wsweep'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/pfix-wsweep-%j-stdout.log
#SBATCH --error=logs/pfix-wsweep-%j-stderr.log

set -e

# Weight sweep (Suspect #4) on the already-written ORACLE_PFIX_DATA column. Down-weights the fill
# in WEIGHT_SPECTRUM to frac x WEIGHT and images ONLY the inpainted column, reusing the clean and
# flagged FITS from the oracle_pfix run (made at full weight / flagged). Tests whether down-weighting
# turns the off-source-RMS win into a fidelity (RMSE-vs-clean) win too. Restores full weight at the end.
SIM=${SIM:-1}
MS=${MS:?set MS=/path/to/sim_clean.ms}
H5=${H5:?set H5=/path/to/dataset.h5}
INPCOL=${INPCOL:-ORACLE_PFIX_DATA}
FRACS=${FRACS:-"0.05 0.2 0.5"}
IMSIZE=${IMSIZE:-2048}
CELL=${CELL:-2asec}
NITER=${NITER:-10000}

ROOT=/users/$USER/rfi-inpainting-research-pipeline
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
OXKAT=/idia/software/containers/$(ls /idia/software/containers/ | grep -i oxkat | head -1)
OUT=${OUT:-/idia/users/$USER/rfi/viz/oracle_pfix}
IMG=$OUT/img
mkdir -p $IMG logs

SIMARG=""; [ "$SIM" = "1" ] && SIMARG="--sim"
if [ ! -f "$IMG/clean-image.fits" ] || [ ! -f "$IMG/flagged-image.fits" ]; then
    echo "missing $IMG/clean-image.fits or flagged-image.fits - run archive/inference/jobs/oracle_phasefix.sh first"; exit 1
fi

wsc () {  # name
    singularity exec $OXKAT wsclean -name $IMG/$1 -data-column $INPCOL \
        -size $IMSIZE $IMSIZE -scale $CELL -niter $NITER -mgain 0.9 \
        -weight briggs -0.3 -auto-mask 4 -auto-threshold 1 -use-wgridder -channels-out 1 \
        -no-update-model-required "$MS"
}

for F in $FRACS; do
    TAG=$(echo $F | tr -d '.')
    echo "==== weight-frac $F -> set WEIGHT_SPECTRUM ===="
    singularity exec $ASTROPY python $ROOT/evaluation/set_holes_weight.py \
        --ms "$MS" --h5 "$H5" --frac $F $SIMARG
    echo "==== image $INPCOL at weight-frac $F ===="
    wsc inp_w$TAG
    echo "==== compare (clean/flagged reused, inpainted = weight-frac $F) ===="
    singularity exec $ASTROPY python $ROOT/evaluation/compare_images.py \
        --clean $IMG/clean-image.fits --flagged $IMG/flagged-image.fits \
        --inpainted $IMG/inp_w$TAG-image.fits --out $OUT/cmp_w$TAG.png
done

echo "==== restore full weight (frac 1.0) ===="
singularity exec $ASTROPY python $ROOT/evaluation/set_holes_weight.py \
    --ms "$MS" --h5 "$H5" --frac 1.0 $SIMARG
echo "done -> $OUT (cmp_w*.png, fits in $IMG/)"
