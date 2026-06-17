#!/bin/bash
# Generate N independent sim runs (different random sky + seed each) toward ~10k baselines.
# Run on the LOGIN node: bash simulate_all.sh   (it generates skies, then sbatches N jobs)
set -e

N_RUNS=${N_RUNS:-5}
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
SIMDIR=$SCRIPTS/data_preparation/simulated
JOB=$SIMDIR/jobs/simulate.sh

for r in $(seq 1 $N_RUNS); do
    SKY=$SIMDIR/sky_random_${r}.txt
    echo "generating sky $r -> $SKY"
    singularity exec $ASTROPY python $SIMDIR/make_random_sky.py --output $SKY --seed $r
    JID=$(sbatch --parsable \
        --export=ALL,RUN_ID=$r,SEED=$((100 + r)),SKY_MODEL=sky_random_${r}.txt \
        $JOB)
    echo "  submitted RUN_ID=$r as job $JID"
done

echo ""
echo "submitted $N_RUNS sim runs. when all done, train on all with:"
echo "  --data '/scratch3/users/\$USER/rfi/simulated/run*/dataset.h5'"
