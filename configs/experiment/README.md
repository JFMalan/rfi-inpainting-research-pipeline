# Experiments and ablations

One YAML per experiment; `bash run_pipeline.sh configs/experiment/<name>.yaml` submits it.
Experiments with a `stages:` list submit only those stages and assume the `final`
experiment's datasets exist on scratch.

| Config | What it runs |
|--------|--------------|
| `final.yaml` | the production experiment: 11 sim runs, phase 1, real extraction, both phase-2 modes, full eval suite (28 stages) |
| `ablation_noisy_target.yaml` | ablation #2, clean-vs-noisy target: phase 1 with the noisy target + sim imaging chain into its own MS column |
| `massoud_r0..r3.yaml` | ablation #1 rungs: R0 = paper recipe (amp-only, raw amps, mixed masking, L2), R1 +div-norm, R2 +phase channels, R3 +noise-free target; each trains on the fixed run1-3 subset and scores the held-out test run via `evaluate_sim` (metrics.json feeds `figures/massoud_ladder.py`). Submit after `final` (they read its datasets) |

## Recipe-style ablations (existing machinery, no dedicated config)

These reuse cluster-tested queue scripts; run them after the `final` experiment produced
its checkpoints. All paths below assume the `final` naming.

**Massoud R4 (sampling/write-back techniques, inference-only)** — the R3 checkpoint
scored with the production sampling stack; compare against R3's own `evaluate_sim`
(50-step, no noise floor) and delay eval:

    sbatch --export=ALL,H5=/scratch3/users/$USER/rfi/simulated/runtest/dataset.h5,CKPT=/idia/users/$USER/rfi/runs/massoud_r3_phase1/best.pt,OUT=/idia/users/$USER/rfi/runs/massoud_r4_eval/fakehole.npz,NOISE_FLOORS="none 0.3 0.5 auto" evaluation/jobs/fakehole_delay.sh

**Ablation #3 (weighted imaging, WEIGHT_FRAC sweep {0, 0.2, 0.5, 1.0})** — the existing
queue chains write-back + imaging per weight:

    bash inference/jobs/downweight_delay_queue.sh    # env-driven; see its header for MS/H5/PREDS

**Ablation #4 (sampling techniques)** — env sweeps on the delay eval:

    sbatch --export=ALL,H5=...,CKPT=...,NOISE_FLOORS="none 0.3 0.5 auto",ETA=1.0,REPAINT_U=2,POST_SAMPLE=1 evaluation/jobs/fakehole_delay.sh

**Ablation #5 (native tiling vs downsample-512)** — real-data variants:

    sbatch data_preparation/real/jobs/extract_variants.sh   # builds the variant h5s
    bash model/real/compare_variants.sh                     # trains/evals the pair

**Ablation #6 (flag-fraction / RFI-width sweep)**:

    bash inference/jobs/rfi_width_sweep.sh                  # inject_width -> infer -> image per width
    # then figures/plot_width_sweep.py on the summary npz

**Ablation #7 (noise generalisation 0x/2x/4x)**:

    bash inference/jobs/lecturer_experiments.sh             # separate MSes per level, runs in parallel
