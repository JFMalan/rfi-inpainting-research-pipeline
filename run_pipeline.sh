#!/bin/bash
# Master orchestrator: submits the whole pipeline for one experiment as an sbatch
# afterok chain and records stage -> jobid in logs/pipeline_<name>.json.
#
#   bash run_pipeline.sh configs/experiment/final.yaml            # submit everything
#   bash run_pipeline.sh configs/experiment/final.yaml --dry-run  # print the plan
#   bash run_pipeline.sh configs/experiment/final.yaml --force train_phase1
#   python3 orchestrator/status.py configs/experiment/final.yaml  # stage/jobid/state table
#
# Rerunning after a failure skips COMPLETED stages, reuses PENDING/RUNNING jobs,
# and resubmits failed ones. Stages stay individually runnable: resolve the env with
#   eval $(python3 orchestrator/resolve_config.py <experiment> --stage <stage> [--run N])
# and sbatch the stage script directly.
set -e
cd "$(dirname "$0")"
if [ -z "$1" ]; then echo "usage: run_pipeline.sh configs/experiment/<name>.yaml [--dry-run] [--only s1,s2] [--force s1,s2]"; exit 1; fi
mkdir -p logs
python3 orchestrator/submit.py "$@"
