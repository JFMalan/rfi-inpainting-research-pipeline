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
NCHAN=${NCHAN:-1024}
SKY_MODEL=${SKY_MODEL:-sky_model.txt}
SEED=${SEED:-42}
NOISE_SCALE=${NOISE_SCALE:-1.0}   # 0 = noise-free; 1 = physical MeerKAT SEFD; 2/4 = higher for a noise sweep
DIR=${DIR:-"J2000,04h00m00.0s,-30d00m00s"}
IMG_SIZE=${IMG_SIZE:-512}
TARGET_FRAC=${TARGET_FRAC:-0.37}
PERSIST_FRAC=${PERSIST_FRAC:-}    # inject_rfi --persist-frac (script default when empty)
RFI_SCALE_MIN=${RFI_SCALE_MIN:-}
RFI_SCALE_MAX=${RFI_SCALE_MAX:-}
SMOOTH_BINS=${SMOOTH_BINS:-}      # extract --smooth-bins (script default when empty)
MAX_BL_FLAG=${MAX_BL_FLAG:-}

# instrument parameters, sourced from configs/telescope/*.yaml via the orchestrator;
# defaults here are the validated MeerKAT L-band values
TEL_MODEL=${TEL_MODEL:-meerkat}
POLS=${POLS:-"XX YY"}
DUMP_T=${DUMP_T:-8}
F0_MHZ=${F0_MHZ:-880}
BW_MHZ=${BW_MHZ:-856.0}
FREQ_MIN=${FREQ_MIN:-900}
FREQ_MAX=${FREQ_MAX:-1650}
SEFD_NODES=${SEFD_NODES:-}        # "mhz:jy mhz:jy ..." for add_noise (built-in profile when empty)

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

# generate this run's random sky on the compute node (singularity is blocked on login)
FLUX_MIN=${FLUX_MIN:-0.1}
FLUX_MAX=${FLUX_MAX:-5.0}
if [ "${GEN_RANDOM_SKY:-0}" = "1" ]; then
    SKY_MODEL=sky_random_${RUN_ID}.txt
    echo "[0/6] $(date '+%H:%M:%S') generating random sky (seed $SEED, flux ${FLUX_MIN}-${FLUX_MAX} Jy) -> $SKY_MODEL"
    singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/make_random_sky.py \
        --output $SCRIPTS/data_preparation/simulated/$SKY_MODEL --seed $SEED \
        --flux-min $FLUX_MIN --flux-max $FLUX_MAX
fi

echo "RUN_ID=$RUN_ID  SYNTHESIS=${SYNTHESIS}h  NCHAN=$NCHAN  SKY_MODEL=$SKY_MODEL  SEED=$SEED"

# compute channel width to keep the configured total bandwidth
DFREQ=$(python3 -c "print(f'{$BW_MHZ/$NCHAN:.4f}MHz')")

echo "[1/6] $(date '+%H:%M:%S') creating empty $TEL_MODEL MS (simms)"
singularity exec $SIMMS simms \
    -T $TEL_MODEL \
    -dir "$DIR" \
    -dt $DUMP_T \
    -st $SYNTHESIS \
    -f0 ${F0_MHZ}MHz \
    -df $DFREQ \
    -nc $NCHAN \
    -pl "$POLS" \
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
    -c $SCRIPTS/data_preparation/simulated/add_noise.py $SIM_MS $SEED $NOISE_SCALE \
    ${SEFD_NODES:+"$SEFD_NODES"} ${SEFD_NODES:+$DUMP_T}

EXTRACT_EXTRA=""
if [ -n "$SMOOTH_BINS" ]; then EXTRACT_EXTRA="$EXTRACT_EXTRA --smooth-bins $SMOOTH_BINS"; fi
if [ -n "$MAX_BL_FLAG" ]; then EXTRACT_EXTRA="$EXTRACT_EXTRA --max-bl-flag-frac $MAX_BL_FLAG"; fi

echo "[4/6] $(date '+%H:%M:%S') extracting native per-baseline waterfalls"
singularity exec $ASTROPY python $SCRIPTS/data_preparation/simulated/extract_patches_sim.py \
    --ms $SIM_MS \
    --output $CLEAN_H5 \
    --waterfall-out $WATERFALL \
    --freq-min $FREQ_MIN --freq-max $FREQ_MAX $EXTRACT_EXTRA

INJECT_EXTRA=""
if [ -n "$PERSIST_FRAC" ]; then INJECT_EXTRA="$INJECT_EXTRA --persist-frac $PERSIST_FRAC"; fi
if [ -n "$RFI_SCALE_MIN" ]; then INJECT_EXTRA="$INJECT_EXTRA --scale-min $RFI_SCALE_MIN"; fi
if [ -n "$RFI_SCALE_MAX" ]; then INJECT_EXTRA="$INJECT_EXTRA --scale-max $RFI_SCALE_MAX"; fi

echo "[5/6] $(date '+%H:%M:%S') injecting synthetic RFI on the native band and tiling to ${IMG_SIZE}x${IMG_SIZE}"
singularity exec $ASTROPY /bin/bash -c "
    source $VENV/bin/activate &&
    python $SCRIPTS/data_preparation/simulated/inject_rfi.py \
        --input $CLEAN_H5 --output $DATASET --img-size $IMG_SIZE --seed $SEED --target-frac $TARGET_FRAC $INJECT_EXTRA
"

echo "[6/6] $(date '+%H:%M:%S') validating dataset and generating visualisations"
singularity exec $ASTROPY python $SCRIPTS/figures/visualise_simulate.py \
    --input $DATASET \
    --output $VISDIR \
    --waterfall ${WATERFALL}.npy \
    --n-plot 12 \
    --n-patches-show 200

echo "done $(date '+%H:%M:%S')"
