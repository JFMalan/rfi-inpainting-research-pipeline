#!/bin/bash
#SBATCH --job-name='rfi-copy-ms'
#SBATCH --partition=Main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/copy-ms-%j-stdout.log
#SBATCH --error=logs/copy-ms-%j-stderr.log

set -e
SRC=${SRC:?set SRC=/path/to/src.ms}
DST=${DST:?set DST=/path/to/dst.ms}
t0=$(date +%s)
echo "copy $(date '+%H:%M:%S')  $SRC ($(du -sh $SRC | cut -f1)) -> $DST"
rm -rf "$DST"
cp -r "$SRC" "$DST"
echo "done $(date '+%H:%M:%S')  $((($(date +%s)-t0)/60)) min  -> $DST ($(du -sh $DST | cut -f1))"
