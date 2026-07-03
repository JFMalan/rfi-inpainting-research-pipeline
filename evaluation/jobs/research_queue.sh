#!/bin/bash
# Submit the research-verdict job sweep. Run from the repo root on ilifu:
#   bash evaluation/jobs/research_queue.sh
# Each block maps to one point of the 2026-07-03 deep-research verdict. Comment out any
# block you don't want. Jobs are independent and queue on SLURM; the location test waits
# on its re-extraction via afterok. Results land in $VIZ as .npz + the SLURM stdout logs
# (the verdict line "MODEL wins both / split" is what to read).

set -e
FT=${FT:-/idia/users/$USER/rfi/runs/phase2_decompose_fullamp/v6_native512_finetune/best.pt}
H5=${H5:-/scratch3/users/$USER/rfi/real/variants/v6_native512.h5}
VIZ=/idia/users/$USER/rfi/viz
Q=evaluation/jobs/fakehole_delay.sh
sub () { echo "  -> $(env H5=$H5 CKPT=$FT "$@" sbatch --parsable $Q)"; }

echo "[3] GPR baseline + noise_floor, ell sweep (mixed holes) -- find GPR's fair setting, is the model's win real"
sub OUT=$VIZ/fd_ell30.npz  NOISE_FLOORS="none 0.5" GPR_ELL=30
sub OUT=$VIZ/fd_ell60.npz  NOISE_FLOORS="none 0.5" GPR_ELL=60
sub OUT=$VIZ/fd_ell120.npz NOISE_FLOORS="none 0.5" GPR_ELL=120

echo "[2] persistent-band ceiling: band-shaped vs blob-shaped holes (both over GOOD pixels, geometry only)"
sub OUT=$VIZ/fd_band.npz NOISE_FLOORS="none 0.5" HOLE_MODE=band GPR_ELL=60
sub OUT=$VIZ/fd_blob.npz NOISE_FLOORS="none 0.5" HOLE_MODE=blob GPR_ELL=60

echo "[4] principled posterior sampling vs flat noise_floor (ensemble mean-of-spectra)"
sub OUT=$VIZ/fd_post_eta.npz     NOISE_FLOORS="none 0.5" POST_SAMPLE=1 ETA=1.0 ENSEMBLE=4 MAX_UNITS=150
sub OUT=$VIZ/fd_post_repaint.npz NOISE_FLOORS="none 0.5" POST_SAMPLE=1 ETA=0 REPAINT_U=4 ENSEMBLE=4 MAX_UNITS=150

echo "[6] divisor test: does divisive-norm distort the delay metric (score in normalised space)"
sub OUT=$VIZ/fd_nodivnorm.npz NOISE_FLOORS="none 0.5" NO_DIVNORM=1 GPR_ELL=60

echo "[5] tighten the bootstrap CI: use all held-out tiles"
sub OUT=$VIZ/fd_alltiles.npz NOISE_FLOORS="none 0.5" GPR_ELL=60 MAX_UNITS=650

echo "[2b] persistent-band LOCATION ceiling: re-extract keeping real flags in the bands, then band-hole eval there"
NOFORCE_DIR=/scratch3/users/$USER/rfi/real/variants_noforce
EJID=$(OUTDIR=$NOFORCE_DIR ONLY=v6_native512 NOFORCE=1 sbatch --parsable data_preparation/real/jobs/extract_variants.sh)
echo "  -> re-extract $EJID"
echo "  -> $(H5=$NOFORCE_DIR/v6_native512.h5 CKPT=$FT OUT=$VIZ/fd_band_location.npz \
       NOISE_FLOORS="none 0.5" HOLE_MODE=band GPR_ELL=60 \
       sbatch --parsable --dependency=afterok:$EJID $Q)"

echo ""
echo "IMAGING GATE (point 1/F) -- separate, only needed before any imaging claim; runs on SIM data."
echo "Fill in your sim clean MS + sim h5 and run:"
echo "  SIM=1 MS=<sim_clean.ms> H5=<sim dataset.h5> sbatch inference/jobs/oracle_level0.sh"
echo "  # if ORACLE0==clean, then: SIM=1 MS=<sim_clean.ms> H5=<sim dataset.h5> sbatch inference/jobs/oracle_phasefix.sh"

echo ""
echo "submitted. watch: squeue -u \$USER ; results: ls -t $VIZ/fd_*.npz ; grep verdict logs/fakehole-delay-*-stdout.log"
