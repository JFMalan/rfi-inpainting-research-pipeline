#!/bin/bash
#SBATCH --job-name='rfi-sim-inpaint-viz'
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100|A40|V100
#SBATCH --qos=qos-interactive
#SBATCH --cpus-per-task=4
#SBATCH --mem=28GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/sim-inpaint-viz-%j-stdout.log
#SBATCH --error=logs/sim-inpaint-viz-%j-stderr.log

set -e

GPU=/idia/software/containers/ASTRO-GPU-PyTorch-2026-01-28.sif
ASTROPY=/idia/software/containers/ASTRO-PY3.10.sif
SCRIPTS=/users/$USER/rfi-inpainting-research-pipeline/model
DATA=${DATA:-/scratch3/users/$USER/rfi/simulated/run1/dataset.h5}
FULL=${FULL:-/idia/users/$USER/rfi/runs/phase1_all/best.pt}                # full-amplitude model
DECOMP=${DECOMP:-/idia/users/$USER/rfi/runs/phase1_all_decompose/best.pt}  # smooth-target model
OUTDIR=/scratch3/users/$USER/rfi/vis-sim
N=${N:-6}; STEPS=${STEPS:-200}; WORST=${WORST:-1}

mkdir -p $OUTDIR logs
LIBCUDA=$(ls /usr/lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null | head -1)
LIBNVML=$(ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* 2>/dev/null | head -1)
NVBIND="--bind $LIBCUDA:/usr/lib/x86_64-linux-gnu/libcuda.so.1 --bind $LIBNVML:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

WARG=""; [ "$WORST" = "1" ] && WARG="--worst"

run_one () {
    local tag=$1 ckpt=$2
    [ -f "$ckpt" ] || { echo "checkpoint missing: $ckpt, skipping $tag"; return; }
    echo "=== inpaint sim [$tag]  ckpt=$ckpt (GT shown = RAW clean) ==="
    singularity exec --nv $NVBIND $GPU python $SCRIPTS/diagnostics/inpaint_viz.py \
        --data $DATA --ckpt $ckpt --out $OUTDIR/sim_${tag}.npz --n $N --steps $STEPS $WARG
    singularity exec $ASTROPY python $SCRIPTS/diagnostics/visualise_samples.py \
        --input $OUTDIR/sim_${tag}.npz --output $OUTDIR/sim_${tag}.png --n-show $N
    echo "  -> $OUTDIR/sim_${tag}.png"
}

run_one full   $FULL      # the original 34x model: should fill fringes SHARP
run_one decomp $DECOMP    # smooth-target model: fills smooth by design

echo "done. compare $OUTDIR/sim_full.png (sharp) vs sim_decomp.png (smooth)"
