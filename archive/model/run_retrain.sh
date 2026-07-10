#!/bin/bash
# Login-node orchestrator for the retrain. Submits sim training (full-amp + decompose)
# and chains the real finetune+scratch to start only after the sim decompose checkpoint
# is written (--dependency=afterok). Pure sbatch — no singularity runs here.
# Run AFTER the sim data (run1..5/dataset.h5) is generated.
set -e

ROOT=/users/$USER/rfi-inpainting-research-pipeline
RUNS=/idia/users/$USER/rfi/runs
INIT=$RUNS/phase1_all_decompose_80ep/best.pt
VAR=/scratch3/users/$USER/rfi/real/variants/v1_upsample512.h5

JID_FULL=$(sbatch --parsable $ROOT/model/sim/train_sim.sh)
echo "sim full-amp          -> job $JID_FULL   (out: $RUNS/phase1_all)"

JID_DEC=$(sbatch --parsable $ROOT/model/sim/train_sim_decompose.sh)
echo "sim decompose         -> job $JID_DEC   (out: $RUNS/phase1_all_decompose_80ep)"

if [ ! -f "$VAR" ]; then
    echo "WARNING: real variant missing: $VAR"
    echo "  the chained real job will fail until you extract real data (flag_real.sh -> extract_variants.sh)."
fi

JID_REAL=$(sbatch --parsable --dependency=afterok:$JID_DEC \
    --export=ALL,INIT=$INIT $ROOT/archive/model/real/finetune_decompose.sh)
echo "real finetune+scratch -> job $JID_REAL   (starts after $JID_DEC; seed INIT=$INIT)"

echo ""
echo "watch: squeue -u \$USER"
echo "the full-amp sim ($JID_FULL) runs independently; real ($JID_REAL) waits on decompose ($JID_DEC)."
