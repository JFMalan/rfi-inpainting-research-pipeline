#!/bin/bash
#SBATCH --job-name='rfi-simulate'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/simulate-%j.log

mkdir -p /scratch3/users/$USER/rfi/simulated logs

singularity exec /idia/software/containers/ASTRO-PY3.simg python data_preparation/simulate.py \
    --output /scratch3/users/$USER/rfi/simulated/dataset.h5 \
    --n_samples 10000 \
    --n_time 256 \
    --n_freq 256 \
    --seed 42

singularity exec /idia/software/containers/ASTRO-PY3.simg python data_preparation/validate_simulate.py \
    --input /scratch3/users/$USER/rfi/simulated/dataset.h5
