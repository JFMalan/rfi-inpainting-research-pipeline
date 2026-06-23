#!/bin/bash
# Generate N independent sim runs (different random sky + seed each) toward ~10k baselines.
# Run on the LOGIN node: bash simulate_all.sh
# Pure submit loop — each job generates its own random sky on the compute node
# (GEN_RANDOM_SKY=1). No singularity runs on the login node (it blocks user namespaces).
set -e

N_RUNS=${N_RUNS:-5}
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline
JOB=$SCRIPTS/data_preparation/simulated/jobs/simulate.sh

for r in $(seq 1 $N_RUNS); do
    JID=$(sbatch --parsable \
        --export=ALL,RUN_ID=$r,SEED=$((100 + r)),GEN_RANDOM_SKY=1 \
        $JOB)
    echo "submitted RUN_ID=$r (random sky seed $r) as job $JID"
done

echo ""
echo "submitted $N_RUNS sim runs. when all done, train on all with:"
echo "  --data '/scratch3/users/\$USER/rfi/simulated/run*/dataset.h5'"
