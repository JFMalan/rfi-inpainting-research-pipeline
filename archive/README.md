# Archive

Superseded scripts kept for provenance. Not part of the active pipeline. Moved here
2026-06-23 during the post-stochastic-inpainting cleanup.

## diagnostics/
| Script | Replaced by | Note |
|--------|-------------|------|
| `bias_diag.py` (+ `jobs/bias_diag.sh`) | `model/diagnostics/pipeline_doctor.py` | intra-hole edge/interior bias is now TEST 4 of the doctor |
| `infer_compare.py` | `model/diagnostics/pipeline_doctor.py` | in-mask vs out-mask noise error is now TEST 5 |
| `gen_sweep.py` | training harness flags (`rand_mask`, `time_roll`, `dropout`) | config ablation absorbed into `config.py` |
| `sampler_sweep.py` (+ `jobs/sampler_sweep.sh`) | `model/diagnostics/stochastic_inpaint.py` | eta/noise-floor sweep + texture ratio supersedes the old eta sweep |
| `viz_eta.py` | `model/diagnostics/stochastic_inpaint.py` | rendered `sampler_sweep.py` output |

## data_preparation/real/
| Script | Replaced by | Note |
|--------|-------------|------|
| `extract_windows.py` | `data_preparation/real/extract_ms.py` | per-baseline extraction replaced the windowed approach |
| `merge_windows.py` | — | only consumed by `extract_windows.py` |

The moved `jobs/*.sh` had their internal script paths repointed to `archive/diagnostics/`,
so they still run if invoked directly, but they are not part of the current workflow.

## 2026-07-10 restructure archival (58 scripts)

Second cleanup pass, ahead of the final.md restructure. Unlike the 2026-06-23 pass, these
files keep their ORIGINAL relative path under `archive/` (e.g. `model/diagnostics/speckle_probe.py`
-> `archive/model/diagnostics/speckle_probe.py`) rather than being flattened. Moved `jobs/*.sh`
had their internal paths repointed to the new `archive/...` locations of their companion scripts
(and, where a moved `.py` imported a sibling module that did NOT move, e.g. `rfi_bands.py`, its
`sys.path` shim was repointed back at the real module) so every file here still runs standalone.

### decompose/smooth-target era
| Script | Replaced by | Note |
|--------|-------------|------|
| `data_preparation/simulated/realify.py` (+ `jobs/decompose_probe.sh`, `jobs/realify_test.sh`) | none - dead end | built the `clean_smooth` decompose target and speckle-calibration variants; smooth/decompose was abandoned in favour of full-amplitude, now the noise-free-target |
| `data_preparation/simulated/visualisation/decompose_layers.py` (used by `data_preparation/real/jobs/recoverable_real.sh`, `jobs/sigma_sweep_real.sh`) | none - dead end | smooth/grain low-pass split that justified the decompose design |
| `model/diagnostics/reversible_inpaint.py` (+ `jobs/reversible_inpaint.sh`) | none - dead end | low/high decomposition + level-match only makes sense for decompose-era checkpoints |
| `model/diagnostics/compare_inpaint.py` (+ `jobs/compare_inpaint.sh`) | `model/diagnostics/compare_models_real.py` | sim/finetune/scratch comparison hardcoded to `phase1_all_decompose` / `phase2_decompose` checkpoints |
| `model/real/finetune_decompose.sh` | none - dead end | `--smooth-target` finetune+scratch comparison |
| `model/real/jobs/eval_real_fix.sh` | `model/real/compare_variants.sh` | fixed decompose-vs-fullamp checkpoint comparison from a resolved audit question |
| `model/run_retrain.sh` | planned config-driven Master Orchestrator | wired specifically to chain the decompose sim checkpoint into `finetune_decompose.sh` |

### speckle/recoverability probes
| Script | Replaced by | Note |
|--------|-------------|------|
| `data_preparation/real/characterise_speckle.py` | none - dead end | real speckle statistics (std ratio, lag-1 autocorr) to calibrate `realify.py` |
| `data_preparation/real/cross_baseline_recoverability.py` (+ `jobs/cross_baseline.sh`) | none - dead end | the recoverability-ceiling probe (ran once 2026-06-24, R2=-0.09); finding is settled, not a repeat-eval stage |
| `data_preparation/real/recover_real.py` (+ `jobs/recover_real.sh`) | none - dead end | behind the "real amplitude ~86% irreducible white noise" finding (speckle_sweep_finding) |
| `data_preparation/real/sim_real_gap.py` (+ `jobs/sim_real_gap.sh`) | none - dead end | sim-vs-real amplitude/PSD/phase/RFI-morphology comparison; explicitly named a dead-end probe |
| `data_preparation/real/visualisation/compare_real_vs_sim.py` | superseded by `sim_real_gap.py` (itself archived) | pre-dataset.h5 era, reads a `.npy`/`.meta.npy` waterfall format nothing produces any more |
| `data_preparation/simulated/jobs/speckle_probe.sh` | none - dead end | speckle_std sweep 0.00-0.18; same "irreducible noise" verdict |
| `data_preparation/simulated/visualisation/vis_speckle.py` | none - dead end | sim-vs-real speckle texture viz, needs a `clean_smooth` field only `realify.py` produces |
| `model/diagnostics/ceiling_check.py` | `model/diagnostics/noise_free_fill_check.py` | smoothed-proxy "noise floor" predates the real paired clean/noisy h5 approach |
| `model/diagnostics/info_ceiling.py` | none - dead end | context->hole R2 / whole-image-PSNR critique on an old-format h5; superseded by hole-region MAE + noise-free-target framing |
| `model/diagnostics/recoverability.py` | `evaluation/classical_fill.py` (DPSS/GPR) | crude mean-fill/interp/biharmonic comparison, superseded by proper classical baselines |
| `model/diagnostics/speckle_probe.py` | none - dead end | the literal speckle probe: target=smooth vs target=noisy |

### TRE-era phase-2 wrappers
| Script | Replaced by | Note |
|--------|-------------|------|
| `model/diagnostics/overfit_real.py` (+ `jobs/overfit_real.sh`) | `model/diagnostics/compare_models_real.py` | phase-2 wiring smoke test scored partly on TRE; default job config never even touches real data |
| `model/diagnostics/jobs/smoke512.sh` | none - dead end | one-off batch-size/timing profiler for the 512-patch upgrade, bundled with a correctness gate; upgrade already shipped |
| `model/diagnostics/smoke512.py` | none - dead end | pure throughput/memory profiler, no PASS/FAIL assertions |
| `model/diagnostics/jobs/sweep.sh` | fixed in `model/config.py` | hyperparameter sweep (hole-fill zero/mean/center, predict noise/x0); its conclusions are now the fixed defaults |
| `model/real/beat_meanfill.py` (+ `jobs/beat_meanfill.sh`) | `model/real/eval_real.py` | tied to the old `variants/*.h5` naming convention final.md says not to reuse; hole-size stratification not in the eval suite |
| `model/real/finetune.sh` | `model/real/compare_variants.sh` | greps TRE columns (dead metric) and trains old `v1_upsample512`/`v4_relaxed512` variants |
| `model/real/train_real.sh` | `model/real/finetune.sh` | earliest single-variant Phase-2 launcher prototype, points at a pre-variants `dataset.h5` path |

### sweep-extract era
| Script | Replaced by | Note |
|--------|-------------|------|
| `data_preparation/real/jobs/sweep_extract.sh` | `data_preparation/real/extract_ms.py` directly | calls `extract_ms.py` with `--sigma-clip`/`--max-flag-frac`, flags that no longer exist (current: `--max-bl-flag-frac`/`--max-ts-flag-frac`) |
| `data_preparation/real/jobs/sweep_compare.sh` | none - dead end | compares the retired sigma-clip/smooth-bins sweep configs |
| `data_preparation/real/jobs/sweep_configs.json` | none - dead end | 8 named param sets for the retired extraction sweep |
| `data_preparation/real/visualisation/compare_sweep.py` | none - dead end | mean-spectrum/flag-fraction comparison across the sweep configs |
| `data_preparation/real/usable_subset.py` | `extract_ms.py --max-bl-flag-frac` | flag-fraction-cutoff picker; the cutoff is now a first-class CLI flag on the extractor |

### imaging bug-hunt oracles
| Script | Replaced by | Note |
|--------|-------------|------|
| `inference/oracle_phasefix.py` (+ `jobs/oracle_phasefix.sh`) | `inference/repr_diag.py` | "Suspect #1" phase-angle-resize test; investigation moved on to `repr_diag.py`'s finer bisection, which is the kept validation gate |
| `inference/jobs/oracle_pfix_wsweep.sh` | `evaluation/set_holes_weight.py` + `inference/jobs/downweight_delay_queue.sh` (production weight sweep) | one-off weight-frac sweep on oracle (non-model) data testing "Suspect #4"; the production weighted-imaging ablation lives in `downweight_delay_queue.sh` |
| `inference/jobs/master_queue.sh` | GROUP A -> a `tests/` oracle-gate runner (planned); GROUP B -> `inference/jobs/selective_inpaint_queue.sh` | GROUP A (level-0 + phasefix oracle gate) and GROUP B (selective inpaint) belong to different buckets; GROUP B duplicates `selective_inpaint_queue.sh` near-verbatim |

### one-off inspections
| Script | Replaced by | Note |
|--------|-------------|------|
| `data_preparation/real/audit_real.py` (+ `jobs/audit_real.sh`) | none - dead end | H1-H5 hypothesis checks explaining a historical amplitude-scale confusion |
| `data_preparation/real/inspect_ms.py` | none - dead end | pre-extraction flag-fraction/dead-baseline sanity check; zero repo references |
| `data_preparation/real/jobs/flag_real_test.sh` | `data_preparation/real/jobs/flag_real.sh` | quick-smoke-test version; calls `extract_ms.py`/`visualise_real.py` with flags (`--max-patches-per-bl`, `--max-time`, `--patches`) that no longer exist |
| `data_preparation/simulated/jobs/simulate_test.sh` | `data_preparation/simulated/jobs/simulate.sh` / `jobs/reextract.sh` | small-scale dry run; calls `extract_patches_sim.py --img-size`, a flag dropped after the native-extract/tile-in-inject refactor |
| `data_preparation/simulated/make_bright_sky.py` (+ `sky_model_bright.txt`) | `data_preparation/simulated/sky_model.txt` | generated the 40x-bright sky; restructure mandates the normal flux range everywhere (see `simulate.sh`/`lecturer_experiments.sh` default changes) |
| `data_preparation/simulated/simulate_vis.yml` | `data_preparation/simulated/jobs/simulate.sh` | early Stimela2/meqtrees prototype of the simulate stage, orphaned once the direct simms/crystalball shell pipeline shipped |
| `model/diagnostics/diagnose_model.py` | none - dead end | ad-hoc in-mask vs out-of-mask noise-prediction probe, zero callers |

### superseded queue/combined-job wrappers
| Script | Replaced by | Note |
|--------|-------------|------|
| `evaluation/jobs/pair_dataset.sh` (+ `evaluation/make_paired_dataset.py`) | native `phase_target` field built into extraction (planned) | retrofit noisy/clean pairing script; final.md: "do not resurrect the retrofit pairing script, build this into extraction" |
| `evaluation/jobs/research_queue.sh` | planned config-driven Master Orchestrator | bespoke one-time sweep tied to the 2026-07-03 deep-research verdict; the jobs it calls (`fakehole_delay.sh` etc.) stay production |
| `inference/jobs/inpaint_ms.sh` | `inference/jobs/inpaint_infer.sh` + `inference/jobs/inpaint_writeback.sh` | combined GPU+CPU stage, self-labelled "UNVALIDATED on real data"; superseded by the split two-stage jobs |

Files still runnable standalone: every moved `.py`/`.sh` above keeps working from its new
`archive/` path (checkpoints/data paths on `/scratch3` or `/idia` are unaffected by this move).
