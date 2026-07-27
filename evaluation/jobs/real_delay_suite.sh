#!/bin/bash
# Paper-grade REAL fake-hole delay suite on the delay-selected checkpoints. Ground-truthed (fake
# holes over known-good real pixels), per-tile spectra saved for bootstrap CIs. Covers: the headline
# finetune-vs-classical comparison, a paired finetune-vs-scratch run (identical seeded holes), hole
# geometry robustness (mixed/blob/band), and a hole-fraction sweep. All jobs are independent GPU
# runs (~30-40 min each) and queue behind whatever holds the fast GPUs.
# Run from repo root:  bash evaluation/jobs/real_delay_suite.sh
set -e
H5=${H5:-/scratch3/users/$USER/rfi/real/v6_native512.h5}
FT=${FT:-/idia/users/$USER/rfi/runs/final_phase2_finetune/best.pt}    # delay-selected e59
SC=${SC:-/idia/users/$USER/rfi/runs/final_phase2_scratch/best.pt}     # delay-selected e47
MAX_UNITS=${MAX_UNITS:-400}
VIZ=/idia/users/$USER/rfi/viz/real_delay_suite
mkdir -p logs
for f in "$H5" "$FT" "$SC"; do [ -e "$f" ] || { echo "missing: $f"; exit 1; }; done

sub() {  # tag ckpt hole_mode "frac_lo frac_hi"
    env H5=$H5 CKPT=$2 OUT=$VIZ/$1.npz MAX_UNITS=$MAX_UNITS HOLE_MODE=$3 FRAC_RANGE="$4" GPR_ELL=30 \
        sbatch --parsable evaluation/jobs/fakehole_delay.sh
}

echo "=== real delay suite -> $VIZ ==="
# headline + paired finetune-vs-scratch (identical seeded holes, mixed, default frac)
J=$(sub ft_mixed   $FT mixed "0.1 0.25"); echo "ft_mixed   $J"
J=$(sub sc_mixed   $SC mixed "0.1 0.25"); echo "sc_mixed   $J  (paired vs ft_mixed)"
# hole-geometry robustness (finetune)
J=$(sub ft_blob    $FT blob  "0.1 0.25"); echo "ft_blob    $J"
J=$(sub ft_band    $FT band  "0.1 0.25"); echo "ft_band    $J  (persistent-band-shaped ceiling test)"
# hole-fraction sweep (finetune, mixed geometry)
J=$(sub ft_frac_mid  $FT mixed "0.2 0.35"); echo "ft_frac_mid  $J"
J=$(sub ft_frac_high $FT mixed "0.4 0.55"); echo "ft_frac_high $J"
echo ""
echo "results: $VIZ/<tag>.npz (per-tile spectra tiles_* for bootstrap CIs); each log has the headline verdict"
