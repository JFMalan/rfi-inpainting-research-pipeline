#!/bin/bash
#SBATCH --job-name='rfi-simulate'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/simulate-%j-stdout.log
#SBATCH --error=logs/simulate-%j-stderr.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jfmalan123@gmail.com

mkdir -p /scratch3/users/$USER/rfi/simulated

singularity exec /idia/software/containers/ASTRO-PY3.10.sif python data_preparation/simulate.py \
    --output /scratch3/users/$USER/rfi/simulated/dataset.h5 \
    --n_samples 10000 \
    --n_time 256 \
    --n_freq 256 \
    --seed 42

singularity exec /idia/software/containers/ASTRO-PY3.10.sif python data_preparation/validate_simulate.py \
    --input /scratch3/users/$USER/rfi/simulated/dataset.h5
