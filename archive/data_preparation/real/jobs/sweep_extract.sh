#!/bin/bash
#SBATCH --job-name='rfi-sweep'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/sweep-%A_%a-stdout.log
#SBATCH --error=logs/sweep-%A_%a-stderr.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

set -e

SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
FLAGGED_MS=/scratch3/users/$USER/rfi/real/1525469431_flagged.ms
CONFIGS=$SCRIPTS/archive/data_preparation/real/jobs/sweep_configs.json
SWEEP_DIR=/scratch3/users/$USER/rfi/sweep
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif

mkdir -p $SWEEP_DIR logs

CFG=$(python3 -c "import json,sys; c=json.load(open('$CONFIGS'))[$SLURM_ARRAY_TASK_ID]; print(c['id'],c['smooth_bins'],c['sigma_clip'],c['max_flag_frac'])")
read -r CFG_ID SMOOTH_BINS SIGMA_CLIP MAX_FLAG_FRAC <<< "$CFG"

OUT_H5=$SWEEP_DIR/${CFG_ID}.h5
VIS_OUT=$SWEEP_DIR/vis_${CFG_ID}

echo "[$SLURM_ARRAY_TASK_ID] id=$CFG_ID smooth=$SMOOTH_BINS sigma=$SIGMA_CLIP max_flag=$MAX_FLAG_FRAC"

singularity exec $ASTROPY python $SCRIPTS/data_preparation/real/extract_ms.py \
    --ms           $FLAGGED_MS \
    --output       $OUT_H5 \
    --freq-min     900 \
    --freq-max     1650 \
    --field        0 \
    --smooth-bins  $SMOOTH_BINS \
    --sigma-clip   $SIGMA_CLIP \
    --max-flag-frac $MAX_FLAG_FRAC

singularity exec $ASTROPY python $SCRIPTS/figures/visualise_real.py \
    --ms       $FLAGGED_MS \
    --output   $VIS_OUT \
    --patches  $OUT_H5 \
    --freq-min 900 \
    --freq-max 1650 \
    --field    0

echo "done -> $VIS_OUT"
