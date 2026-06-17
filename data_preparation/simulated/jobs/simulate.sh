#!/bin/bash
#SBATCH --job-name='rfi-simulate'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/simulate-%j-stdout.log
#SBATCH --error=logs/simulate-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

# Defaults — override via: sbatch --export=ALL,RUN_ID=2,SYNTHESIS=2.0,NCHAN=4096,SKY_MODEL=sky_model_2.txt simulate.sh
RUN_ID=${RUN_ID:-1}
SYNTHESIS=${SYNTHESIS:-1.2}
NCHAN=${NCHAN:-4096}
SKY_MODEL=${SKY_MODEL:-sky_model.txt}
SEED=${SEED:-42}
IMG_SIZE=${IMG_SIZE:-512}
TARGET_FRAC=${TARGET_FRAC:-0.40}

SIMMS=/idia/software/containers/STIMELA_IMAGES/stimela_simms_1.2.0.sif
AFRICANUS=/idia/software/containers/STIMELA_IMAGES/stimela_codex-africanus_1.6.7.sif
CASA=/idia/software/containers/casa-stable-v6.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
VENV=/idia/users/$USER/venvs/rfi_toolbox
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline

SIMDIR=/scratch3/users/$USER/rfi/simulated/run${RUN_ID}
SIM_MS=$SIMDIR/sim_clean.ms
WATERFALL=$SIMDIR/waterfall
CLEAN_H5=$SIMDIR/clean_baselines.h5
DATASET=$SIMDIR/dataset.h5
VISDIR=$SIMDIR/vis

mkdir -p $SIMDIR $VISDIR logs
rm -rf $SIM_MS

echo "RUN_ID=$RUN_ID  SYNTHESIS=${SYNTHESIS}h  NCHAN=$NCHAN  SKY_MODEL=$SKY_MODEL  SEED=$SEED"

# compute channel width to keep 856 MHz total bandwidth
DFREQ=$(python3 -c "print(f'{856.0/$NCHAN:.4f}MHz')")

echo "[1/6] $(date '+%H:%M:%S') creating empty MeerKAT MS (simms)"
singularity exec $SIMMS simms \
    -T meerkat \
    -dir "J2000,04h00m00.0s,-30d00m00s" \
    -dt 8 \
    -st $SYNTHESIS \
    -f0 880MHz \
    -df $DFREQ \
    -nc $NCHAN \
    -pl "XX YY" \
    -n $SIM_MS

echo "[2/6] $(date '+%H:%M:%S') predicting sky model (crystalball)"
singularity exec $AFRICANUS crystalball \
    -sm $SCRIPTS/data_preparation/simulated/$SKY_MODEL \
    -o DATA \
    -rc 10000 \
    -mc 25 \
    -j 8 \
    $SIM_MS

echo "[3/6] $(date '+%H:%M:%S') adding thermal noise (CASA sm.corrupt)"
singularity exec $CASA casa --nologger --log2term \
    -c $SCRIPTS/data_preparation/simulated/add_noise.py $SIM_MS

echo "[4/6] $(date '+%H:%M:%S') extracting per-baseline ${IMG_SIZE}x${IMG_SIZE} waterfalls"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/extract_patches_sim.py \
    --ms $SIM_MS \
    --output $CLEAN_H5 \
    --waterfall-out $WATERFALL \
    --freq-min 900 --freq-max 1650 \
    --img-size $IMG_SIZE

echo "[5/6] $(date '+%H:%M:%S') injecting synthetic RFI"
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate &&
    python $SCRIPTS/data_preparation/simulated/inject_rfi.py \
        --input $CLEAN_H5 --output $DATASET --seed $SEED --target-frac $TARGET_FRAC
"

echo "[6/6] $(date '+%H:%M:%S') validating dataset and generating visualisations"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/visualisation/visualise_simulate.py \
    --input $DATASET \
    --output $VISDIR \
    --waterfall ${WATERFALL}.npy \
    --n-plot 12 \
    --n-patches-show 200

echo "done $(date '+%H:%M:%S')"
