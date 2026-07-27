#!/bin/bash
#SBATCH --job-name='rfi-copy-ms'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/copy-ms-%j-stdout.log
#SBATCH --error=logs/copy-ms-%j-stderr.log

set -e

# The source real MS lives on read-only /idia; the write-back stages need a writable
# copy on scratch (INPAINTED_DATA / GPR_DATA columns get added to it).
SRC_MS=${SRC_MS:?set SRC_MS=/path/to/source.ms}
DST_MS=${DST_MS:?set DST_MS=/scratch3/.../copy.ms}

mkdir -p $(dirname $DST_MS) logs

if [ -d "$DST_MS" ]; then
    echo "destination exists, leaving as-is: $DST_MS"
    exit 0
fi

echo "$(date '+%H:%M:%S') copying $SRC_MS -> $DST_MS"
cp -r $SRC_MS $DST_MS
echo "$(date '+%H:%M:%S') done ($(du -sh $DST_MS | cut -f1))"
