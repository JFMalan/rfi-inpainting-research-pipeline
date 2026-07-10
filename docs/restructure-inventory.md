# Restructure inventory — keep vs archive map

Generated 2026-07-10 from the pre-restructure tree (tag `pre-restructure`, commit 766d12c),
classified per final.md criteria by a fan-out review, cross-checked, and hand-corrected.
Classifications: **production** (pipeline simulate→train→extract→finetune→inpaint→write-back→evaluate,
incl. ablation machinery), **validation** (correctness gates → move to tests/), **figure** (report
figure generators → feed figures/), **archive** (one-offs, dead ends, superseded eras).

Counts: production 61, validation 8, figure 14, archive 58 — total 141


## PRODUCTION (61)

### `data_preparation/real/extract_ms.py`
Core real-MS -> per-baseline h5 extractor: reads DATA column in chunks, computes amplitude/phase/flags per (time,baseline), splits into contiguous good-timestamp runs, applies divisive normalisation, resizes each (run,baseline) waterfall to img-size, force-flags persistent RFI bands, writes data/phase/flags/dn_divisor + patch-position metadata to h5.
- evidence: archive/README.md states extract_ms.py replaced extract_windows.py ('per-baseline extraction replaced the windowed approach'); its current CLI signature (--ms/--output/--column/--freq-min/--freq-max/--img-size/--max-bl-flag-frac/--max-ts-flag-frac/--min-run/--smooth-bins/--force-persistent) is exactly what jobs/extract_real.sh and jobs/flag_real.sh call; this is final.md's 'extract real' pipeline stage.
- invokes: data_preparation/real/rfi_bands.py
- invoked by: data_preparation/real/jobs/extract_real.sh, data_preparation/real/jobs/flag_real.sh, data_preparation/real/jobs/flag_real_test.sh (stale flags), data_preparation/real/jobs/sweep_extract.sh (stale flags)
- note: flag_real_test.sh and sweep_extract.sh call it with --max-patches-per-bl/--max-time or --sigma-clip/--max-flag-frac, none of which exist in this file's current argparse -- those two job scripts are stale/broken against the current script, confirming they are archive-era leftovers even though extract_ms.py itself is current production.

### `data_preparation/real/extract_variants.py`
Builds several named real-data dataset variants (v1_upsample512 .. v6_native512) from one MS in a single pass: whole-band upsample, time-windowed, freq-tiled, relaxed/loose flag thresholds, and v6_native512 which does native-resolution overlapping freq tiles (via tiling.py) matching the sim extractor's tiling.
- evidence: v6_native512 (the default ONLY=v6_native512 in extract_variants.sh) implements the native freq-tiling strategy that MEMORY project_tiling_impl / project_resolution_research names as the adopted resolution-fix ('freq tiling+overlap reuse 512 model'); imports the shared tiling.py and rfi_bands.py used by production extractors.
- invokes: data_preparation/real/rfi_bands.py, data_preparation/tiling.py (freq_tile_starts/freq_tile_width/time_extent)
- invoked by: data_preparation/real/jobs/extract_variants.sh
- note: The legacy v1-v5 variant specs in this same file (upsample/time-windowed/freq-tiled-2/relaxed) are earlier iterations kept in the same function for comparison; only v6_native512 is the current default, but classifying the whole file as production since it is the live real-data extractor script and v6 is reachable via --only.

### `data_preparation/real/jobs/extract_real.sh`
SLURM job (Main, 8h, 128GB) running extract_ms.py on the production target MS (1570802018 J2018_5539) at freq 900-1650 MHz, img-size 512, --max-bl-flag-frac override via MAX_BL_FLAG env, writing dataset.h5 to scratch.
- evidence: Its flags (--ms/--output/--column/--freq-min/--freq-max/--img-size/--max-bl-flag-frac) match extract_ms.py's current argparse exactly -- this is the live extract-real job.
- invokes: data_preparation/real/extract_ms.py

### `data_preparation/real/jobs/extract_variants.sh`
SLURM job (Main, 4h, 128GB) running extract_variants.py, defaulting ONLY=v6_native512 (the tiled variant), with env-overridable MAXTSFLAG/MINRUN/MAXFLAG/NOFORCE (the last enabling a 'location ceiling test' that keeps tricolour flags in persistent bands instead of force-flagging them).
- evidence: Default ONLY=v6_native512 comment says explicitly 'tiled variant matching the sim extractor'; this is the current real-data extraction job for the native-tiling strategy.
- invokes: data_preparation/real/extract_variants.py

### `data_preparation/real/jobs/flag_real.sh`
SLURM job (Main, 10h, 128GB, 32 cpus) that copies the source MS to scratch, runs tricolour with tricolour-flagging.yaml, then calls extract_ms.py (--field 0) and visualise_real.py (with a --patches flag) to extract patches and visualise the flagged MS.
- evidence: This is the actual tricolour flagging job (production RFI-flagging stage per CLAUDE.md 'tricolour is the flagger'); its extract_ms.py call uses only current-valid flags (--ms/--output/--freq-min/--freq-max/--field).
- invokes: tricolour (oxkat container), data_preparation/real/tricolour-flagging.yaml, data_preparation/real/extract_ms.py, data_preparation/real/visualisation/visualise_real.py
- note: The final visualise_real.py call passes --patches $PATCHES_OUT, but visualise_real.py's current argparse has no --patches flag -- this call would fail with 'unrecognized arguments'; a real discrepancy to fix in the restructure, but the tricolour + extract_ms.py portions of the job are current and correct.

### `data_preparation/real/rfi_bands.py`
Defines LBAND_PERSISTENT_MHZ (MeerKAT L-band static/persistent RFI frequency ranges) and persist_chan_mask() helper.
- evidence: Imported and actively used (force-flagging persistent bands) by the current production extractors extract_ms.py and extract_variants.py, and by cross_baseline_recoverability.py and visualisation/compare_sweep.py; this is core shared config data for the extract-real pipeline stage.
- invoked by: data_preparation/real/extract_ms.py, data_preparation/real/extract_variants.py, data_preparation/real/cross_baseline_recoverability.py, data_preparation/real/visualisation/compare_sweep.py
- note: Comment in the persistent-band list ('MeerKAT Cookbook short-baseline emitter list... absorbed by tricolour's background estimator') documents it is meant to complement, not replace, tricolour flagging.

### `data_preparation/real/tricolour-flagging.yaml`
tricolour RFI-flagging strategy config: nan/zero dropout flag, static background mask, sum_threshold background flagging, uvcontsub_flagger residual flagging (two passes), a second static mask at uvrange 0~550, three sum_threshold passes at decreasing spike widths, flag_autos, combine_with_input_flags.
- evidence: CLAUDE.md states 'tricolour is the flagger' and this is the only tricolour config file in the repo; it is passed via --config to the tricolour CLI invocation in jobs/flag_real.sh and jobs/flag_real_test.sh, i.e. it is the actual RFI-flagging step of the extract-real pipeline stage.
- invoked by: data_preparation/real/jobs/flag_real.sh, data_preparation/real/jobs/flag_real_test.sh

### `data_preparation/simulated/README.md`
Documents the two-stage sim data-prep pipeline (extract_patches_sim.py -> inject_rfi.py) and the dataset.h5 schema/attrs used for MS write-back.
- evidence: Describes clean/corrupted/mask/phase/dn_divisor/position-metadata schema that model/ and inference/write-back code consume; still the reference doc for the active pipeline.
- note: Stale in one detail: it says extract_patches_sim.py tiles into 256x256 patches, but the current extract_patches_sim.py only produces native per-baseline waterfalls (no tiling, no img-size arg) — tiling now lives in inject_rfi.py via tiling.py (matches memory note 'Tiling implemented': native-512 freq tiles built into inject-before-tile). README needs a small rewrite for the restructure, not a rewrite of the pipeline itself. Also needs updating for the planned phase_target (clean-phase) field from final.md.

### `data_preparation/simulated/add_noise.py`
CASA script that adds MeerKAT-SEFD-shaped thermal noise (flat sm.corrupt + per-channel residual) to a simulated MS; noise_scale=0 instead copies DATA to CORRECTED_DATA for a noise-free extraction.
- evidence: Invoked by jobs/simulate.sh step [3/6] and jobs/simulate_test.sh; the noise_scale=0 DATA->CORRECTED_DATA landmine is explicitly called out in final.md ('keep that behaviour in any refactor').
- invoked by: data_preparation/simulated/jobs/simulate.sh, data_preparation/simulated/jobs/simulate_test.sh
- note: final.md's noise-free-target plan (0.7-1.0x SEFD draw, snapshot pre-noise DATA for a clean target) will need this script extended/reused, but the existing noise_scale=0 mechanics are exactly what that plan requires.

### `data_preparation/simulated/extract_patches_sim.py`
Reads a simulated MS, builds per-baseline amplitude/phase waterfalls with divisive normalisation, and writes native (un-tiled) clean_baselines.h5 with position metadata.
- evidence: Invoked by jobs/simulate.sh [4/6], jobs/simulate_test.sh [4/6], jobs/reextract.sh [4/5] — the extraction stage of simulate -> ... -> write-back.
- invoked by: data_preparation/simulated/jobs/simulate.sh, data_preparation/simulated/jobs/simulate_test.sh, data_preparation/simulated/jobs/reextract.sh
- note: Only extracts amplitude+phase from the single (noisy) MS column — final.md's headline noise-free-target plan requires this script to additionally snapshot pre-noise DATA and emit a phase_target field sharing the noisy divisor; that is new/extended work, not yet present. jobs/simulate_test.sh calls it with --img-size, a flag this script does not define — a stale caller, not evidence the script itself is stale.

### `data_preparation/simulated/inject_rfi.py`
Loads clean_baselines.h5, injects synthetic RFI (stochastic rfi_toolbox generator or a deterministic controlled-width mode), tiles the native band into overlapping (default 512) freq/time patches via tiling.py, and writes dataset.h5.
- evidence: Invoked by jobs/simulate.sh [5/6], jobs/simulate_test.sh [5/6], jobs/reextract.sh [5/5], and jobs/inject_width.sh (controlled band-width mode for the rfi_width_sweep ablation named in final.md ablation #6).
- invokes: data_preparation/tiling.py (freq_tile_starts/freq_tile_width/time_extent), data_preparation/real/rfi_bands.py (LBAND_PERSISTENT_MHZ), rfi_toolbox.data_generation.synthetic_generator.SyntheticDataGenerator
- invoked by: data_preparation/simulated/jobs/simulate.sh, data_preparation/simulated/jobs/simulate_test.sh, data_preparation/simulated/jobs/reextract.sh, data_preparation/simulated/jobs/inject_width.sh
- note: controlled_spans/controlled_inject (--band-width flag) is exactly the rfi_width_sweep machinery final.md lists as production ablation #6 — same script serves both the main pipeline and that ablation.

### `data_preparation/simulated/jobs/inject_width.sh`
Controlled RFI-width test: runs inject_rfi.py's deterministic band-width mode over an existing clean_baselines.h5 at a fixed target flag fraction, producing a dataset.h5 that drops straight into inpaint_infer/image_eval.
- evidence: Own header comment states the output schema matches inject_rfi.py's normal schema for direct use by inpaint_infer/image_eval; this is the rfi_width_sweep ablation final.md lists (#6 'existing rfi_width_sweep machinery').
- invokes: data_preparation/simulated/inject_rfi.py
- note: None.

### `data_preparation/simulated/jobs/reextract.sh`
Re-runs extraction+injection (steps 4-5) on an already-simulated sim_clean.ms without re-running simms/crystalball/add_noise — used to regenerate dataset.h5 for an existing run, e.g. after a tiling/injection change.
- evidence: Own header comment: 'steps 4-5 of simulate.sh... same SEED reproduces the RFI'; calls the same production extract_patches_sim.py / inject_rfi.py pair.
- invokes: data_preparation/simulated/extract_patches_sim.py, data_preparation/simulated/inject_rfi.py
- note: None.

### `data_preparation/simulated/jobs/simulate.sh`
Full production simulate-stage job: simms -> crystalball predict -> CASA add_noise -> extract_patches_sim -> inject_rfi -> visualise_simulate, parameterised by RUN_ID/SYNTHESIS/NCHAN/SKY_MODEL/SEED/NOISE_SCALE/IMG_SIZE/TARGET_FRAC.
- evidence: Is the canonical 'simulate' stage of the simulate -> train -> ... pipeline named in final.md; every sub-script it calls is itself classified production.
- invokes: simms (STIMELA_IMAGES container), crystalball (africanus container), data_preparation/simulated/add_noise.py, data_preparation/simulated/extract_patches_sim.py, data_preparation/simulated/inject_rfi.py, data_preparation/simulated/visualisation/visualise_simulate.py, data_preparation/simulated/make_random_sky.py (conditional on GEN_RANDOM_SKY=1)
- invoked by: data_preparation/simulated/jobs/simulate_all.sh (via sbatch)
- note: Default SKY_MODEL is sky_model_bright.txt (the 40x-bright one) — final.md explicitly says the restructured pipeline should default to the normal sky_model.txt flux range instead; this script's default will need updating even though the script itself is production.

### `data_preparation/simulated/jobs/simulate_all.sh`
Login-node submit loop that sbatches N independent simulate.sh runs (each with its own random sky via GEN_RANDOM_SKY=1 and a distinct seed) toward the multi-run training set.
- evidence: Directly implements final.md's 'Produce 10 training runs + 1 held-out test run' requirement as a batch submitter over the production simulate.sh job.
- invokes: data_preparation/simulated/jobs/simulate.sh (via sbatch)
- note: final.md's master orchestrator will likely subsume/replace this loop, but the capability (multi-run submission) is production and reusable, not to be rewritten.

### `data_preparation/simulated/make_random_sky.py`
Generates a random point-source sky model (count and flux drawn per run) at the sim field centre, for per-run sky diversity.
- evidence: final.md explicitly names it as reused as-is: 'random sky per run (existing make_random_sky.py)'; invoked conditionally (GEN_RANDOM_SKY=1) by jobs/simulate.sh.
- invoked by: data_preparation/simulated/jobs/simulate.sh (conditional GEN_RANDOM_SKY=1), data_preparation/simulated/jobs/simulate_all.sh (indirectly, via simulate.sh)
- note: None.

### `data_preparation/simulated/sky_model.txt`
Normal-flux (0.003-0.12 Jy range) point-source sky model, the base list make_bright_sky.py scales up from.
- evidence: final.md explicitly directs using 'the current normal sky_model.txt range, NOT the old 40x bright one' as the flux scale going forward.
- note: None.

### `data_preparation/tiling.py`
Shared tiling utilities: freq_tile_starts/freq_tile_width compute overlapping frequency-tile offsets for a native band, feather_weight computes partition-of-unity blend weights for seam-free write-back, time_extent computes a center-crop or pass-through time window.
- evidence: Imported by inject_rfi.py (patch tiling), inference/inpaint_write.py (feathered write-back), and data_preparation/real/extract_variants.py (real-data tiling) — a core cross-cutting production utility per the 'native-res-tiling' work (memory: 'Tiling implemented').
- invoked by: data_preparation/simulated/inject_rfi.py, inference/inpaint_write.py, data_preparation/real/extract_variants.py
- note: Its __main__ block is a self-test (prints coverage/partition-of-unity checks for several N/T values) — arguably a validation gate for the tiling math, but it's a debug harness inside the production module rather than a separate script, so the file as a whole stays production.

### `evaluation/classical_fill.py`
Library of the two classical gap-fill baselines: ridge-regularised DPSS least-squares fit and constant-mean GPR (SE kernel) regression, both grouped by identical gap-pattern for efficiency.
- evidence: final.md lists DPSS and GPR (constant-mean) as required classical-fill comparators in the Evaluation suite and Ablations; imported by delay_spectrum.py, dpss_fill_write.py and fakehole_delay_eval.py, all of which are production.
- invoked by: evaluation/delay_spectrum.py, evaluation/dpss_fill_write.py, evaluation/fakehole_delay_eval.py

### `evaluation/compare_images.py`
Loads FITS continuum images (clean/flagged/meanfill/classical/inpainted), computes off-source RMS, peak, dynamic range and RMSE-vs-clean, renders a side-by-side comparison figure and writes a metrics.json.
- evidence: final.md Evaluation suite requires continuum RMSE-vs-clean + dynamic range metrics per variant; invoked by evaluation/image_eval.sh which is the production imaging pipeline; last touched 2026-07-06 for the width sweep.
- invoked by: evaluation/image_eval.sh

### `evaluation/delay_spectrum.py`
Reads clean/flagged/DPSS/inpainted visibilities per baseline from the MS + h5 hole mask, computes tapered FFT delay-power spectra, wlogP-RMSE and hi-delay-ratio metrics vs clean, and plots the comparison.
- evidence: final.md: 'the delay spectrum' is the headline evaluation arena; invoked by both evaluation/image_eval.sh and evaluation/jobs/delay_confirm.sh.
- invokes: evaluation/classical_fill.py
- invoked by: evaluation/image_eval.sh, evaluation/jobs/delay_confirm.sh

### `evaluation/dpss_fill_write.py`
Writes the DPSS classical gap-fill into a new MS data column (per contiguous time/baseline run) as a continuum-imaging baseline, optionally clearing FLAG at filled holes.
- evidence: final.md explicitly cites this file as the pattern to replicate ('GPR MS write-back module analogous to dpss_fill_write.py'), i.e. it names it a production capability; invoked by image_eval.sh (DPSSFILL=1) and evaluation/jobs/width_sweep_summary.sh's upstream width-sweep runs.
- invokes: evaluation/classical_fill.py
- invoked by: evaluation/image_eval.sh

### `evaluation/evaluate.py`
Runs the trained diffusion model over a sim dataset split, samples in-hole fills via diffusion.sample, and reports amp MAE / PSNR / phase error / complex MAE, saving example batches + metrics.json.
- evidence: model/README.md: 'The reportable result comes from evaluation/evaluate.py on the [held-out test split]'; invoked by evaluation/jobs/eval.sh.
- invokes: model/config.py, model/data.py, model/diffusion.py, model/unet.py, model/metrics.py
- invoked by: evaluation/jobs/eval.sh

### `evaluation/fakehole_delay_eval.py`
The ground-truthed real-data delay test: injects fake holes over known-good real pixels, fills with the model (sweeping noise_floor/posterior sampling), DPSS, and GPR, scores wlogP-RMSE/hi-ratio vs true good data with a tile-level bootstrap CI vs the stronger classical baseline.
- evidence: final.md Ablations #4 names this exact module ('existing fakehole_delay_eval variants'); this is the 'only ground-truthed real delay test' per its own header comment; invoked by evaluation/jobs/fakehole_delay.sh, orchestrated at scale by research_queue.sh.
- invokes: model/config.py, model/data.py, model/diffusion.py, model/unet.py, evaluation/classical_fill.py
- invoked by: evaluation/jobs/fakehole_delay.sh, evaluation/jobs/research_queue.sh (via fakehole_delay.sh)

### `evaluation/image_eval.sh`
SLURM job that images Clean/Flagged/Meanfill/DPSSfill/Inpainted MS columns with wsclean, then runs compare_images.py (continuum) and delay_spectrum.py (delay space) for the full write-back comparison; supports KEEP_PERSIST selective-inpaint imaging.
- evidence: Implements exactly the 'BOTH arenas: continuum wsclean image ... and delay spectrum' comparison final.md specifies for the Evaluation suite, including the keep-persist selective-inpaint variant flagged as production in the task's known facts.
- invokes: evaluation/set_holes_flag.py, evaluation/mean_fill_write.py, evaluation/dpss_fill_write.py, evaluation/compare_images.py, evaluation/delay_spectrum.py

### `evaluation/jobs/delay_confirm.sh`
Standalone SLURM job running delay_spectrum.py on the fully written-back INPAINTED_DATA (actual RFI holes, not fake holes), given its own longer walltime because image_eval's delay step used to time out.
- evidence: Directly invokes the production delay_spectrum.py against the real write-back MS; comment: 'End-to-end delay-space confirmation ... this is valid regardless of the WEIGHT_FRAC used at write time.'
- invokes: evaluation/delay_spectrum.py

### `evaluation/jobs/eval.sh`
SLURM job wrapper that runs evaluation/evaluate.py against a phaseN_runN sim checkpoint/dataset with GPU driver-lib binding.
- evidence: Only invokes evaluate.py, the production sim-eval script; sets up the ASTRO-GPU-PyTorch container per the documented landmine in final.md.
- invokes: evaluation/evaluate.py

### `evaluation/jobs/fakehole_delay.sh`
SLURM job wrapper for evaluation/fakehole_delay_eval.py, exposing noise_floor/GPR-ell/hole-mode/posterior-sampling knobs as env vars.
- evidence: Wraps the production fakehole_delay_eval.py; also invoked programmatically by evaluation/jobs/research_queue.sh's sweep.
- invokes: evaluation/fakehole_delay_eval.py
- invoked by: evaluation/jobs/research_queue.sh

### `evaluation/jobs/width_sweep_summary.sh`
SLURM job that runs plot_width_sweep.py to summarise per-RFI-width metrics.json files (image RMSE-vs-clean, dynamic range) into the flag-fraction/width crossover figure.
- evidence: final.md Ablations #6 names this exact machinery ('existing rfi_width_sweep machinery with the final model ... the sweep maps the crossover'), matching the task's explicit criterion that rfi_width_sweep feeds report ablations; also invoked from inference/jobs/lecturer_experiments.sh.
- invokes: evaluation/plot_width_sweep.py
- invoked by: inference/jobs/lecturer_experiments.sh

### `evaluation/mean_fill_write.py`
Writes a per-channel, per-baseline time-mean fill of the hole regions into a new MS column, for a mean-fill continuum-imaging baseline.
- evidence: Invoked by evaluation/image_eval.sh (MEANFILL=1) as one of the standard 3-way (flagged/meanfill/inpainted) continuum benchmark columns that final.md's Evaluation suite requires.
- invoked by: evaluation/image_eval.sh

### `evaluation/plot_width_sweep.py`
Reads per-RFI-width metrics.json (flagged/inpainted/DPSS RMSE-vs-clean and dynamic range) and plots the continuum-fidelity and dynamic-range curves vs width/noise, marking the inpaint-vs-flag crossover.
- evidence: final.md Ablations #6 names 'existing rfi_width_sweep machinery' and the Figures pipeline lists 'the flag-fraction crossover plot' as required; invoked by evaluation/jobs/width_sweep_summary.sh, itself called from lecturer_experiments.sh.
- invoked by: evaluation/jobs/width_sweep_summary.sh

### `evaluation/set_holes_flag.py`
Sets or clears the MS FLAG column at hole locations (all / persistent-band-only / non-persistent-band-only), used to prepare an MS for imaging a given variant (truth/flagged/filled, or selective keep-persist inpaint).
- evidence: Implements the keep-persist selective-inpaint machinery the task's known facts call production; invoked repeatedly by evaluation/image_eval.sh for both the standard flagged-image step and the KEEP_PERSIST selective variant.
- invokes: data_preparation/real/rfi_bands.py (persist_chan_mask)
- invoked by: evaluation/image_eval.sh

### `evaluation/set_holes_weight.py`
Adds/uses WEIGHT_SPECTRUM and down-weights filled hole pixels to a fraction of the baseline weight (WEIGHT_FRAC), clearing FLAG there, for the weighted-imaging ablation.
- evidence: final.md Ablations #3 names this exact module: 'existing set_holes_weight / WEIGHT_FRAC: sweep {0, 0.2, 0.5, 1.0} on sim and real, continuum metric'; invoked by inference/jobs/oracle_pfix_wsweep.sh.
- invoked by: inference/jobs/oracle_pfix_wsweep.sh

### `inference/inpaint_infer.py`
GPU inference script: loads a phase1/phase2 config + checkpoint, runs the diffusion sampler over every unit in an h5 dataset (sim or real), and saves amplitude+cos+sin predictions to a .npz.
- evidence: Implements the core 'inpaint' pipeline stage (simulate->train->extract->finetune->inpaint->write-back->evaluate) and is called by every currently-used job queue via inpaint_infer.sh (lecturer_experiments, rfi_width_sweep, noise_free_image_test, master_queue, selective_inpaint_queue).
- invokes: model/config.py, model/data.py, model/diffusion.py, model/unet.py
- invoked by: inference/jobs/inpaint_infer.sh, inference/jobs/inpaint_ms.sh
- note: Its --oracle flag bypasses the model to write ground truth (used by validation-flavored callers), but the script's own capability is production inference.

### `inference/inpaint_write.py`
CPU write-back script: resizes preds to native resolution, feather-blends overlapping freq tiles, reconstructs the complex visibility from amp*divisor*exp(i*atan2(sin,cos)), and writes it into an MS column with optional weight-frac downweight, unflag, and keep-persist-flagged behaviour.
- evidence: Implements the write-back stage plus the 'weighted imaging (set_holes_weight/WEIGHT_FRAC)' and 'selective inpaint (keep-persist)' capabilities the classification criteria explicitly names as production.
- invokes: data_preparation/tiling.py (feather_weight), data_preparation/real/rfi_bands.py (persist_chan_mask)
- invoked by: inference/jobs/inpaint_writeback.sh, inference/jobs/inpaint_ms.sh, inference/jobs/oracle_phasefix.sh
- note: Also reused by oracle_phasefix.sh (an archive-classified diagnostic) purely as a generic MS-column writer — classified here by the module's own capability, not that caller, per the task instructions.

### `inference/jobs/downweight_delay_queue.sh`
Orchestrator: sweeps WEIGHT_FRAC in {0.2,0.3,0.5} doing write-back + continuum imaging for each, then confirms the delay-space win on the final written INPAINTED_DATA column.
- evidence: Directly implements final.md ablation #3 'Weighted imaging ... sweep {0,0.2,0.5,1.0} on sim and real, continuum metric.'
- invokes: inference/jobs/inpaint_writeback.sh, evaluation/image_eval.sh, evaluation/jobs/delay_confirm.sh
- note: Top-level, run manually; requires PREDS already computed by a prior selective-inpaint infer run.

### `inference/jobs/inpaint_infer.sh`
SLURM GPU job (stage 1 only): wraps inpaint_infer.py with env-var flags (SIM, TAG/CKPT, SMOOTH, NOISE_FLOOR, STEPS, BATCH, MAX_UNITS) and a footgun guard against using a sim checkpoint on real data.
- evidence: Universally used inference stage across every current production queue (lecturer_experiments, rfi_width_sweep, noise_free_image_test, master_queue, selective_inpaint_queue all call it).
- invokes: inference/inpaint_infer.py
- invoked by: inference/jobs/lecturer_experiments.sh, inference/jobs/rfi_width_sweep.sh, inference/jobs/noise_free_image_test.sh, inference/jobs/master_queue.sh, inference/jobs/selective_inpaint_queue.sh

### `inference/jobs/inpaint_writeback.sh`
SLURM CPU job (Main partition, 128GB, stage 2 only): wraps inpaint_write.py with env flags OUTCOL/UNFLAG/KEEP_PERSIST/WEIGHT_FRAC/RESET_COL/NO_FEATHER.
- evidence: Universally used write-back stage — called by every current production queue (downweight_delay_queue, lecturer_experiments, master_queue, noise_free_image_test, rfi_width_sweep, selective_inpaint_queue).
- invokes: inference/inpaint_write.py
- invoked by: inference/jobs/downweight_delay_queue.sh, inference/jobs/lecturer_experiments.sh, inference/jobs/master_queue.sh, inference/jobs/noise_free_image_test.sh, inference/jobs/rfi_width_sweep.sh, inference/jobs/selective_inpaint_queue.sh

### `inference/jobs/lecturer_experiments.sh`
Runs two experiments in the sim domain: (1) noise-generalisation sweep across NOISE_SCALES (0/1/2/4x SEFD, same sky) evaluated with the existing sim model; (2) feathered-tile-blend on/off ablation at the in-distribution noise level.
- evidence: Matches final.md ablation #7 'Noise generalisation — existing lecturer-test machinery: final model evaluated at 0x/2x/4x SEFD' near-verbatim, and the classification criteria explicitly lists 'noise-generalisation (lecturer) tests' as production.
- invokes: data_preparation/simulated/jobs/simulate.sh, inference/jobs/inpaint_infer.sh, inference/jobs/inpaint_writeback.sh, evaluation/image_eval.sh, evaluation/jobs/width_sweep_summary.sh
- note: Top-level, run manually.

### `inference/jobs/noise_free_image_test.sh`
Images Clean/Flagged/Inpaint/DPSS on a noise-free MS using the phase1_thr_paired (noise-free-target) model, testing whether filling RFI holes with the recovered fine structure beats flagging when the imaging reference is perfectly clean.
- evidence: Directly tests the 'noise-free clean-target recipe' the project memory and final.md call the headline finding (ablation #2, 'Clean vs noisy target ... this is a headline result').
- invokes: inference/jobs/inpaint_infer.sh, inference/jobs/inpaint_writeback.sh, evaluation/image_eval.sh
- note: Top-level, run manually; supports NODELIST to pin the GPU infer job.

### `inference/jobs/rfi_width_sweep.sh`
Controlled RFI-band-width sweep (widths 4-128 channels) in the sim domain: injects deterministic stripes, infers, writes back, images vs clean/flagged, and produces a crossover summary plot.
- evidence: Matches final.md ablation #6 'Flag-fraction / RFI-width sweep — existing rfi_width_sweep machinery with the final model' verbatim, including the crossover framing mentioned there.
- invokes: data_preparation/simulated/jobs/inject_width.sh, inference/jobs/inpaint_infer.sh, inference/jobs/inpaint_writeback.sh, evaluation/image_eval.sh, evaluation/jobs/width_sweep_summary.sh
- note: Top-level, run manually; supports SKIP_PHASE1 to reuse prior datasets/preds.

### `inference/jobs/selective_inpaint_queue.sh`
Fills only the non-persistent RFI on a real MS (leaves wide persistent bands flagged) via infer+write-back, then images Flagged-everything vs Selective-inpaint (+DPSS).
- evidence: Implements the 'Inpainted — selective' evaluation variant and the 'keep-persist selective inpaint machinery' the known project facts explicitly call production.
- invokes: model/diagnostics/jobs/compare_models_real.sh, inference/jobs/inpaint_infer.sh, inference/jobs/inpaint_writeback.sh, evaluation/image_eval.sh
- note: Near-duplicate of master_queue.sh's GROUP B (same env vars and job chain) — worth consolidating during the restructure rather than keeping both.

### `model/README.md`
Documents the model directory: 3-channel (amp+cos+sin phase) conditional DDPM, data.h5 input contract, phase-1 training/eval commands, GPU/container notes.
- evidence: Describes the current architecture (config.py in_channels, build_cond, PatchDataset schema) that train.py/train_sim.sh still use unmodified.
- note: Phase 2 section is stale: it says fake-mask injection, sampler tweak and metrics.tre 'need to be implemented' but train_real.py/data.py.RealDataset/metrics.tre already implement all three. Needs a rewrite pass during restructure (final.md explicitly calls for README rewrite) but the Phase-1 content is accurate and current.

### `model/config.py`
Config dataclass + phase1()/phase2() factories: channel counts, U-Net shape, diffusion/loss knobs, mixed-masking and early-stop settings.
- evidence: Imported by train.py, train_real.py, eval_real.py, beat_meanfill.py — every current training/eval entry point builds its Config from here.
- invoked by: model/train.py, model/train_real.py, model/real/eval_real.py, model/real/beat_meanfill.py
- note: Carries vestigial smooth_target/smooth_sigma fields from the now-superseded decompose recipe; harmless (default False) but should be dropped when decompose is purged per final.md ('never a target going forward').

### `model/data.py`
PatchDataset (sim) and RealDataset (real) HDF5 loaders, positional encoding, fake_mask mixed-masking generator, build_cond conditioning assembly, smooth_component low-pass helper.
- evidence: PatchDataset/RealDataset/build_cond/fake_mask are imported by every current train/eval script (train.py, train_real.py, eval_real.py, beat_meanfill.py); this is the shared data layer on the simulate->train->finetune pipeline.
- invoked by: model/train.py, model/train_real.py, model/real/eval_real.py, model/real/beat_meanfill.py
- note: smooth_component() (decompose/decompose-target helper) is exactly the function final.md calls out by name as 'never a target ... it is circular' — this one function inside an otherwise-production file is dead weight to be deleted, not the file as a whole.

### `model/diagnostics/compare_models_real.py`
Samples multiple named checkpoints (sim/finetune/scratch) on real held-out tiles filtered by flag fraction, with keep-persist support to leave persistent RFI bands flagged, and renders the comparison grid.
- evidence: jobs/compare_models_real.sh is invoked directly from inference/jobs/master_queue.sh and inference/jobs/selective_inpaint_queue.sh (the live selective-inpaint production queue) as its viz step; implements persist_chan_mask/--keep-persist, the selective-inpaint machinery final.md explicitly calls production.
- invokes: config.phase2, data.positional_encoding/build_cond, diffusion.Diffusion, unet.UNet, data_preparation/real/rfi_bands.persist_chan_mask
- invoked by: model/diagnostics/jobs/compare_models_real.sh, inference/jobs/master_queue.sh, inference/jobs/selective_inpaint_queue.sh
- note: Also functions as a figure-generator (report selective-inpaint panels) but its wiring into the live SLURM queue makes production the primary classification.

### `model/diagnostics/inpaint_real.py`
Runs a single checkpoint's sampler against the REAL RFI mask (actual flags, not fake holes) on real MeerKAT tiles and dumps an npz for rendering.
- evidence: Invoked by jobs/inpaint_viz.sh under 'REAL inpaint of ACTUAL RFI flags (same sim model)' — this is literally the inpaint pipeline stage applied to real data with true RFI flags, feeding the downstream visualisation/evaluation step.
- invokes: config.phase2, data.positional_encoding/build_cond, diffusion.Diffusion, unet.UNet
- invoked by: model/diagnostics/jobs/inpaint_viz.sh
- note: Distinct from inpaint_viz.py: this one always fills the real flags (no fake-hole scoring), i.e. the genuine production inpaint step on real data.

### `model/diagnostics/jobs/compare_models_real.sh`
SLURM job wrapping compare_models_real.py: sim/finetune/scratch amplitude-inpaint comparison on real held-out tiles, with a KEEP_PERSIST toggle for selective-inpaint visualisation.
- evidence: Directly invoked by inference/jobs/master_queue.sh and inference/jobs/selective_inpaint_queue.sh as the viz step of the live selective-inpaint production chain.
- invokes: model/diagnostics/compare_models_real.py (--h5, --ckpts sim=/finetune=/scratch=, --noise-floor, --keep-persist, --min/max-flag-frac)
- invoked by: inference/jobs/master_queue.sh, inference/jobs/selective_inpaint_queue.sh
- note: Defaults to the current production checkpoints (phase1_all_tiled80ep, phase2_decompose_fullamp v6_native512 finetune/scratch).

### `model/diagnostics/jobs/noise_free_fill_check.sh`
SLURM job wrapping noise_free_fill_check.py against a paired noisy/noise-free dataset and checkpoint (both required, no defaults).
- evidence: Chained afterok from model/sim/jobs/noise_free_target_test.sh (the noise-free-target training orchestrator) — the headline recipe's validation/eval step.
- invokes: model/diagnostics/noise_free_fill_check.py (--h5, --ckpt, --output, --steps, --n-show)
- invoked by: model/sim/jobs/noise_free_target_test.sh
- note: H5 and CKPT are required env vars (no defaults), consistent with being called parametrically from the orchestrator rather than run standalone.

### `model/diagnostics/jobs/stochastic_inpaint.sh`
SLURM job wrapping stochastic_inpaint.py's eta/noise_floor ablation sweep; TAG selects the checkpoint (defaults to phase1_all_decompose).
- evidence: Wraps stochastic_inpaint.py, which implements final.md's sampling-technique ablation (#4) directly; classified by capability per instructions despite the stale default checkpoint.
- invokes: model/diagnostics/stochastic_inpaint.py (--ckpt, --data, --n, --steps, --predict x0, --out-png)
- note: Discrepancy: default TAG=phase1_all_decompose is a superseded decompose-era checkpoint; the job needs its default repointed at a current full-amp model, not its .py rewritten.

### `model/diagnostics/noise_free_fill_check.py`
The headline noise-free-target check: samples a model trained on a paired noisy-context/noise-free-target dataset, compares smooth fill (noise_floor=none) vs matched-grain fill vs true target, and reports delay-spectrum power ratios.
- evidence: jobs/noise_free_fill_check.sh is triggered afterok from model/sim/jobs/noise_free_target_test.sh, the orchestrator for the clean-target training recipe; directly matches final.md ablation #2 ('Clean vs noisy target' — the headline result) and the 'fill-check panels' figure spec.
- invokes: config.phase1, data.positional_encoding/build_cond, diffusion.Diffusion, unet.UNet
- invoked by: model/diagnostics/jobs/noise_free_fill_check.sh, model/sim/jobs/noise_free_target_test.sh
- note: This is the concrete implementation behind the memory note 'Noise-free-target test' (2026-07-08 result); central to the project's current headline finding.

### `model/diagnostics/stochastic_inpaint.py`
Systematically compares sampling conditions (eta in {0,1} x noise_floor in {none, auto}) against mean-fill/freq-interp baselines, reporting texture ratio and MAE vs both the noisy and smooth targets.
- evidence: This is exactly final.md's ablation #4, 'Sampling techniques ... noise_floor {none, 0.3, 0.5, matched} ... eta' — a report ablation, not a one-off probe.
- invokes: config.phase1, data.positional_encoding/build_cond/smooth_component, diffusion.Diffusion, unet.UNet
- invoked by: model/diagnostics/jobs/stochastic_inpaint.sh
- note: Discrepancy: jobs/stochastic_inpaint.sh defaults TAG=phase1_all_decompose, a superseded decompose-era checkpoint — the job's default config is stale, but the .py capability (eta/noise_floor sweep vs baselines) is the production ablation machinery and should simply be repointed at a current full-amp checkpoint.

### `model/diffusion.py`
Cosine-schedule DDPM: q_sample, hole-only masked loss (loss/loss_phase2), DDIM/RePaint masked sampling with optional noise_floor grain-matching.
- evidence: sample()'s noise_floor + repaint_u machinery is exactly the 'sampling techniques' ablation (final.md #4: noise_floor {none,0.3,0.5,matched}, DDIM steps, repaint_u) and the matched-grain-vs-smooth write-back distinction final.md requires per arena.
- invoked by: model/train.py, model/train_real.py, model/real/eval_real.py, model/real/beat_meanfill.py

### `model/metrics.py`
mae/psnr/phase_error/complex_mae/noise_floor_ratio (amplitude+complex-vis metrics) and tre() (Total Reconstruction Error for real data with no clean truth).
- evidence: complex_mae is the metric train.py/train_real.py actually early-stop and checkpoint on ('best.pt' saved on lowest complex_mae); noise_floor_ratio backs the texture-consistency claims in final.md's sampling ablation.
- invoked by: model/train.py, model/train_real.py, model/real/eval_real.py, model/real/beat_meanfill.py
- note: tre() itself is a holdover from the 'old TRE metric era' final.md names as an archive example — it is still computed/printed by train_real.py and eval_real.py for informational logging but no longer drives any decision (early-stop/best.pt use complex_mae). Recommend deleting tre() specifically when its callers are cleaned up; the rest of the file is current.

### `model/real/compare_variants.sh`
Trains train_real.py on every real-data variant .h5 in a directory to an equal iteration budget, evaluates each with eval_real.py, and ranks them by TRE/fake-MAE.
- evidence: final.md ablation #5 names exactly this: 'Native tiling vs downsample-512 — existing real-data variants comparison.'
- invokes: model/train_real.py, model/real/eval_real.py
- note: Will need its dataset glob repointed at the re-extracted config-driven variants and the smooth-target/decompose knobs (none present here, already full-amp) left as-is.

### `model/real/eval_real.py`
Held-out-test real-data eval: samples the model (with selectable noise_floor mode), computes TRE, fake-hole MAE vs interp-baseline vs mean-fill, fill-std ratio and noise-floor ratio.
- evidence: final.md: 'the eval scripts should pick the correct mode per arena without manual switching' for smooth-vs-matched-grain noise_floor — this script already exposes exactly that switch (--noise-floor none|auto|float) as the mechanism to build that automatic per-arena selection on top of.
- invokes: model/config.py, model/data.py, model/diffusion.py, model/unet.py, model/metrics.py
- invoked by: model/real/finetune.sh, model/real/finetune_decompose.sh, model/real/compare_variants.sh, model/real/jobs/eval_real_fix.sh
- note: Reports tre() (dead metric era) alongside fake-MAE/interp/mean-fill/noise-floor-ratio (current); tre print line is vestigial but the rest of the script is the production real-eval capability. Also accepts a --smooth-target passthrough purely for decompose-ckpt parity, which should be dropped with the decompose purge.

### `model/sim/jobs/noise_free_target_test.sh`
Simulates a noise-free twin of an existing noisy sim run, pairs clean-target with noisy-input, trains, then runs a fill-continuity check (delay-space grain matching) — the noise-free-target validation.
- evidence: This is exactly final.md ablation #2, 'Clean vs noisy target — already demonstrated... productionize as a first-class ablation on the final datasets. This is a headline result.'
- invokes: data_preparation/simulated/jobs/simulate.sh, evaluation/jobs/pair_dataset.sh, model/sim/train_sim.sh, model/diagnostics/jobs/noise_free_fill_check.sh
- note: Currently a one-off test script (paired/'thr_paired' naming, ad-hoc pairing via evaluation/jobs/pair_dataset.sh) but final.md says this pairing must move INTO the extraction pipeline natively rather than staying a retrofit ('Do not resurrect the retrofit pairing script; build this into extraction') — so this exact script's pairing mechanism is slated to be replaced even though the ablation/capability it proves stays production.

### `model/sim/jobs/noise_threshold_sweep.sh`
Sweeps thermal-noise scale (1.0/0.5/0.25/0.125x SEFD) on a fixed sky, retraining a model from scratch at each level, to find the SNR threshold where the model stops beating mean-fill.
- evidence: Its finding (recoverability ceiling / trainable-noise range) plausibly underlies final.md's chosen '0.7-1.0x SEFD' training noise range, but final.md's named ablations list does not include a repeat-this-sweep item, and it is not one of the fixed evaluation-suite variants (flagged/DPSS/GPR/inpainted).
- invokes: data_preparation/simulated/jobs/simulate.sh, model/sim/train_sim.sh, evaluation/jobs/noise_threshold_summary.sh
- note: Leaning archive (one-off diagnostic that already produced its verdict, feeding a design decision now baked into final.md) but flagging as unsure since it could be re-run once on the final 10-run dataset as supporting evidence for the noise-range choice; ask before archiving if the report needs to show this curve. | Cross-check: generates the noise-threshold/recoverability curve final.md requires as a report figure.

### `model/train.py`
Phase-1 supervised training loop on simulated dataset.h5: EMA, val-eval against a mean-fill baseline, complex-MAE-based early stop/checkpointing.
- evidence: final.md names it explicitly: 'Phase 1 (sim): existing train.py on the 10 runs. Val metrics already include the mean-fill baseline (amp_mf, beats_mf) — keep that.'
- invokes: model/config.py, model/data.py, model/diffusion.py, model/unet.py, model/metrics.py
- invoked by: model/sim/train_sim.sh, model/sim/train_sim_decompose.sh, model/run_retrain.sh (indirectly, via those two job scripts)
- note: --smooth-target flag support (decompose recipe) should be dropped per final.md; the trainer's core loop is otherwise exactly what phase 1 keeps.

### `model/train_real.py`
Phase-2 mixed-masking real-data trainer: fake-hole self-supervised loss, sim-checkpoint init (--init-from), TRE/complex-MAE val-eval, EMA sized for short fine-tunes.
- evidence: final.md: 'Phase 2 (real MeerKAT): always train BOTH configs — finetune seeded from the sim model AND from-scratch... keep the existing fake-hole self-supervised recipe (noisy target) unchanged' — this is that trainer.
- invokes: model/config.py, model/data.py, model/diffusion.py, model/unet.py, model/metrics.py
- invoked by: model/real/train_real.sh, model/real/finetune.sh, model/real/finetune_decompose.sh, model/real/compare_variants.sh
- note: --smooth-target/--smooth-sigma path (decompose) is dead per final.md and should be stripped; the fake-hole/finetune-vs-scratch/EMA-sizing core is the production capability, invoked today mostly by decompose-era or pre-restructure job scripts (see discrepancy note on those .sh files).

### `model/unet.py`
Conditional U-Net: timestep embedding, residual blocks, self-attention at configurable resolutions, down/up sampling.
- evidence: Instantiated identically by train.py, train_real.py, eval_real.py, beat_meanfill.py with cfg.in_channels/target_channels/base/ch_mult from config.py — the model backbone for the whole pipeline.
- invoked by: model/train.py, model/train_real.py, model/real/eval_real.py, model/real/beat_meanfill.py


## VALIDATION (8)

### `evaluation/hole_pred_check.py`
Directly compares saved model predictions (preds .npz) against the h5 clean/phase/divisor ground truth inside the hole mask, reporting amplitude bias/MAE, phase MAE and complex relative error to localize whether an artifact is a model bias or a write-back bug.
- evidence: final.md explicitly names hole_pred_check as one of the validation gates to move to tests/, and its own commit message says it exists 'to localize sim continuum artifact (model bias vs bug)'.

### `inference/jobs/oracle_level0.sh`
SLURM job that builds the level-0 native-passthrough oracle (writes the exact native DATA at hole pixels, no h5 resize/pol-collapse round trip) and images it against clean/flagged to verify the h5-unit-to-MS-row/channel/pol write-back mapping.
- evidence: final.md explicitly names 'the level-0 oracle write-back test' as a validation gate to be moved to tests/, not archived.
- invokes: inference/oracle_level0.py, evaluation/image_eval.sh
- invoked by: inference/jobs/master_queue.sh

### `inference/oracle_level0.py`
Verifies the h5-unit-to-MS-row/channel/pol mapping (constant baseline antennas, monotonic time, channel alignment) then writes the true native DATA into the hole pixels of an oracle column for imaging comparison.
- evidence: final.md explicitly names 'the level-0 oracle write-back test' as a validation gate that moves to tests/, not archive.
- invoked by: inference/jobs/oracle_level0.sh
- note: Standalone script using casacore/h5py/skimage only.

### `inference/repr_diag.py`
Representation diagnostics: measures visibility-domain RMS error in the RFI holes for amp-resize roundtrip, phase h5-angle vs cos/sin-fix, full pipeline (current vs phase-fixed), and polarisation-collapse, all relative to a per-pol-mean truth built directly from native MS DATA.
- evidence: final.md explicitly names 'the representation diagnostics' as one of the validation gates to keep and move to tests/; docs/methodology-audit.md and docs/tiling-design-brief.md cite its measured numbers (pol-collapse 1.6%, amp-resize roundtrip 0.115) as the ongoing correctness reference.
- note: No SLURM job script in the repo wraps it — appears to be run ad hoc/manually (docs say 'Re-run repr_diag.py to confirm...'); the restructure will likely need to add a tests/ job wrapper for it.

### `model/diagnostics/jobs/overfit.sh`
SLURM job wrapping overfit_test.py as a standalone correctness gate on a chosen simulated run's dataset.
- evidence: Runs overfit_test.py (a PASS/FAIL correctness gate, see that file's entry) with no sweep/comparison intent — a single-config sanity check.
- invokes: model/diagnostics/overfit_test.py (--data, --n, --iters, --bs, --lr, --predict, --hole-fill, --eta, --amp-only)
- note: Data path is /scratch3/.../simulated/run${RUN_ID}/dataset.h5 — the standard sim dataset location.

### `model/diagnostics/overfit_test.py`
Phase-1 training/sampling contract gate: leak-free single-shot in-hole MAE, full-sampler amplitude/PSNR/texture vs mean-fill and freq-interp baselines, complex-visibility MAE, explicit PASS/FAIL verdict.
- evidence: Produces an explicit PASS/FAIL verdict on whether 'the model beats mean-fill on amplitude AND complex' and is used as a standalone correctness gate in jobs/overfit.sh and as part of the combined smoke test in jobs/smoke512.sh.
- invokes: config.phase1, data.PatchDataset/build_cond, diffusion.Diffusion, unet.UNet, metrics.mae/psnr/phase_error/complex_mae
- invoked by: model/diagnostics/jobs/overfit.sh, model/diagnostics/jobs/smoke512.sh, model/diagnostics/jobs/sweep.sh
- note: Also reused by jobs/sweep.sh purely as a hyperparameter-comparison tool (hole-fill zero/mean/center, predict noise/x0) — that specific caller is archived-era since those configuration decisions are now fixed, but the .py's own capability is a genuine correctness gate, so classified by capability per instructions.

### `model/diagnostics/pipeline_doctor.py`
Six-part correctness gate for the whole phase-1 training contract: data integrity, conditioning-leak check, U-Net receptive field, x_in noising contract, loss-grades-the-hole check, and a 2-patch overfit capacity check — all with explicit PASS/FAIL lines.
- evidence: Matches final.md's 'a validation gate proving pipeline correctness' description almost verbatim, ending in 'Read the FAIL lines above to localise the issue'; not tied to any superseded target or metric.
- invokes: config.phase1, data.PatchDataset/build_cond, diffusion.Diffusion, unet.UNet
- note: No .sh wrapper currently exists for it in the repo, but it is the closest match in this file set to the named validation gates (oracle write-back / hole_pred_check / repr_diag) — a strong 'move to tests/' candidate.

### `model/diagnostics/smoke_test.py`
Minimal end-to-end wiring check: dataset shapes match config, conditioning channel count is correct, forward+loss+backward succeeds, small-T sampling runs, ending in 'ALL CHECKS PASSED'.
- evidence: A textbook smoke test that gates correctness (per the given classification criteria) rather than an exploratory diagnostic; distinct from and simpler than smoke512.py's profiling role.
- invokes: config.phase1, data.PatchDataset/build_cond, diffusion.Diffusion, unet.UNet, metrics.mae/psnr/phase_error
- note: No current .sh wrapper found in the repo, but its content is exactly the 'smoke tests that gate correctness' validation category.


## FIGURE (14)

### `data_preparation/real/visualisation/visualise_real.py`
Loads amp/flags/freqs directly from an MS, produces flag-fraction-per-channel/baseline/time diagnostics, per-baseline waterfall grid (green-overlay flags), amplitude distribution, mean spectrum (RFI bands annotated), full averaged waterfall, and zoomed spectrum panels around known missed-RFI regions.
- evidence: Self-consistent, currently-functional MS-diagnostic figure generator (its own argparse --ms/--output/--column/--field/--max-time/--n-baselines/--dpi/--freq-min/--freq-max/--vmax is internally coherent and produces genuinely useful flagging QA plots); classified by capability rather than by its callers.
- invoked by: data_preparation/real/jobs/flag_real.sh (stale --patches arg), data_preparation/real/jobs/flag_real_test.sh (stale --patches arg), data_preparation/real/jobs/sweep_extract.sh (stale --patches arg)
- note: Every caller in this file set passes a --patches flag that does not exist in this script's argparse, so all three job invocations would currently fail with an argument error -- a real discrepancy between caller and capability (per task instructions, classifying by capability: this is a still-useful flagging-QA figure script that needs its callers repaired, not archived code).

### `data_preparation/simulated/visualisation/visualise_simulate.py`
Runs correctness/sanity checks on a generated dataset.h5 (flag-fraction range, NaN checks, RFI-amplitude-at-mask sanity, per-channel/time flag morphology, run-length check with PASS/FAIL) and produces the dataset example figures (sample pairs, amplitude/spectra distributions, full waterfall, patch-page panels).
- evidence: Invoked as the final step [6/6] of both jobs/simulate.sh and jobs/simulate_test.sh; its plots (clean/corrupted/mask sample panels, patch pages) are exactly final.md's required figure 'dataset examples per run (input / RFI mask / clean target)'.
- invoked by: data_preparation/simulated/jobs/simulate.sh, data_preparation/simulated/jobs/simulate_test.sh
- note: Dual-purpose: run_checks() also acts as a lightweight correctness gate (WARNING/PASSED-FAILED on flag fraction, NaNs, RFI morphology) run automatically after every sim run, so it partly overlaps the 'validation' category too — but its primary current role and its explicit invocation for figure generation puts it in figure.

### `evaluation/jobs/noise_threshold_summary.sh`
SLURM job that runs plot_noise_threshold.py to summarise the per-noise-scale trainability sweep (model vs mean-fill amp MAE, PSNR) across a set of phase1_thr_n* runs.
- evidence: final.md's Figures pipeline explicitly lists 'the noise-threshold/recoverability curve' as a required figure; this job (2026-07-07) generates exactly that.
- invokes: evaluation/plot_noise_threshold.py

### `evaluation/plot_delay_npz.py`
Renders the publication-style two-panel figure (log delay-power spectra + power/truth ratio panel with optional bootstrap-CI annotation) from a fakehole_delay_eval.py output .npz.
- evidence: Commit message: 'add plot_delay_npz.py: publication figure from fakehole delay .npz (valid real delay result)'; matches final.md's Figures pipeline requirement for 'delay-spectrum comparisons'; no wrapping job script found in-repo, so it is likely run ad hoc against each research_queue.sh .npz output.

### `evaluation/plot_noise_data.py`
Plots the same training tile's raw clean/corrupted field across a set of thermal-noise scales (runthr_n* dataset.h5 files), for visual comparison of how noise obscures the signal.
- evidence: final.md's Figures pipeline requires 'dataset examples per run'; commit dated 2026-07-08, same investigation as the noise-threshold/recoverability curve; no invoking job script was found in the repo (likely run ad hoc), which is noted as a gap.

### `evaluation/plot_noise_samples.py`
Renders per-noise-scale training-sample panels (observed / truth / model fill, with the hole mask overlaid in green) alongside best-epoch amp MAE / mean-fill / beats-mf annotations from each run's log.json.
- evidence: final.md's Figures pipeline requires 'fill-check panels (observed / target / smooth fill / matched-grain fill / delay spectra)'; commit dated 2026-07-08, part of the current noise-threshold/noise-free-target investigation; no invoking job script found (likely ad hoc).

### `evaluation/plot_noise_threshold.py`
Reads best-epoch amp MAE (model vs mean-fill) and PSNR from each phase1_thr_n* run's log.json across a thermal-noise-scale sweep, plots trainability vs noise and finds the model-vs-mean-fill crossover threshold.
- evidence: final.md's Figures pipeline lists 'the noise-threshold/recoverability curve' verbatim as a required figure; invoked by evaluation/jobs/noise_threshold_summary.sh.
- invoked by: evaluation/jobs/noise_threshold_summary.sh

### `inference/viz_writeback.py`
Plots a 4-panel amplitude waterfall (source column, hole mask, inpainted column, diff) per baseline, comparing an MS's source vs write-back data column for visual QA.
- evidence: Matches final.md's figures-pipeline item 'fill-check panels (observed / target / smooth fill / matched-grain fill / delay spectra)' — its src/hole/inpainted/diff panel layout is exactly that kind of figure, and it is generic (works via --sim flag and any --inp-col), not tied to the superseded smooth/decompose era.
- note: Orphaned in the current repo — no .sh or .py anywhere calls it, so it is presumably invoked by hand; the restructure should give it a job wrapper if it is kept as a figures/ script.

### `model/diagnostics/inpaint_viz.py`
Generic sampler for both sim (clean/mask) and real (fake-hole) data; supports a legacy --smooth-target flag; dumps an npz consumed by visualise_samples.py, plus --worst hardest-hole tile selection.
- evidence: Invoked by jobs/inpaint_viz.sh (sim), jobs/sim_inpaint_viz.sh (still passes FULL=phase1_all/best.pt, the current production sim model), and jobs/regen_real_viz.sh — the core sampling+npz-export capability is still exercised with current checkpoints, only the --smooth-target branch is a dead code path from the superseded decompose era.
- invokes: config.phase1/phase2, data.positional_encoding/fake_mask/build_cond/smooth_component, diffusion.Diffusion, unet.UNet
- invoked by: model/diagnostics/jobs/inpaint_viz.sh, model/diagnostics/jobs/sim_inpaint_viz.sh, model/diagnostics/jobs/regen_real_viz.sh
- note: Discrepancy noted: file still contains smooth_component/--smooth-target plumbing for the superseded decompose era, but is also the live figure-generation path for current full-amp production checkpoints.

### `model/diagnostics/jobs/inpaint_viz.sh`
SLURM job running sim inpaint (inpaint_viz.py) and real actual-flag inpaint (inpaint_real.py) with one checkpoint, then rendering both with visualise_samples.py / visualise_real_inpaint.py.
- evidence: Defaults CKPT to the current production sim model phase1_all/best.pt; entire job's output is two report PNGs (sim_inpaint.png, real_inpaint.png).
- invokes: model/diagnostics/inpaint_viz.py (--data, --ckpt, --out, --n), model/diagnostics/inpaint_real.py (--data, --ckpt, --out, --n), model/diagnostics/visualise_samples.py (--input, --output, --n-show), model/diagnostics/visualise_real_inpaint.py (--input, --output, --n-show)
- note: Mixes a genuine production inference step (inpaint_real.py) with figure rendering; classified by its dominant deliverable (two report figures).

### `model/diagnostics/jobs/regen_real_viz.sh`
SLURM job re-rendering the real-data inpaint figure for the current full-amp sim checkpoint, with an optional (DECOMPOSE=1) legacy decompose-checkpoint comparison.
- evidence: Primary path uses SIM_CKPT=phase1_all/best.pt (current production model) and produces real_phase1_all.png; the DECOMPOSE=1 branch calling inpaint_viz.py --smooth-target against phase2_decompose/v1_upsample512_finetune is the only archived-era part.
- invokes: model/diagnostics/inpaint_viz.py (--data, --ckpt, --out, --real, --n, [--smooth-target]), model/diagnostics/visualise_samples.py (--input, --output, --n-show)
- note: Discrepancy: the file is mostly current-figure but retains a superseded optional branch (DECOMPOSE=1) — worth stripping in the restructure rather than archiving the whole job.

### `model/diagnostics/jobs/sim_inpaint_viz.sh`
SLURM job running inpaint_viz.py + visualise_samples.py twice: once for the current full-amp model (FULL=phase1_all/best.pt) and once for the legacy decompose model (DECOMP=phase1_all_decompose/best.pt), for a side-by-side sharp-vs-smooth comparison.
- evidence: Comment explicitly frames it as 'compare sim_full.png (sharp) vs sim_decomp.png (smooth)' — a deliberate current-vs-superseded comparison figure, still useful to show why the decompose approach was dropped.
- invokes: model/diagnostics/inpaint_viz.py (--data, --ckpt, --out, --n, --steps, --worst), model/diagnostics/visualise_samples.py (--input, --output, --n-show)
- note: One of its two model arguments (DECOMP) points at a superseded checkpoint, but the job itself is a current, meaningful comparison figure, not a dead end.

### `model/diagnostics/visualise_real_inpaint.py`
Renders an inpaint_real.py npz (observed vs inpainted amplitude+phase on the actual real RFI mask) into report-style panel grids.
- evidence: Invoked by jobs/inpaint_viz.sh as the final render step for the real-data inpaint demonstration.
- invoked by: model/diagnostics/jobs/inpaint_viz.sh
- note: Companion renderer to visualise_samples.py, specialised for the real-flags npz schema (data/phase/flags/pred rather than clean/corrupted/mask/pred).

### `model/diagnostics/visualise_samples.py`
Renders an inpaint_viz.py npz (clean/corrupted/pred/mask, amplitude+phase) into report-style comparison panel grids.
- evidence: Invoked by jobs/inpaint_viz.sh, jobs/sim_inpaint_viz.sh, jobs/regen_real_viz.sh — a generic, still-current rendering utility over any predict-mode output.
- invoked by: model/diagnostics/jobs/inpaint_viz.sh, model/diagnostics/jobs/sim_inpaint_viz.sh, model/diagnostics/jobs/regen_real_viz.sh
- note: Pure matplotlib rendering script; no model/data logic of its own.


## ARCHIVE (58)

### `data_preparation/real/audit_real.py`
Loads a real per-baseline h5 (default v1_upsample512.h5) and runs 5 hypothesis checks (H1-H5) on amplitude scale/outliers, spatial structure of outliers, mean-fill vs interp recoverability raw/clipped, per-channel scale reconciliation, and DN-divisor sanity.
- evidence: One-off hypothesis-testing diagnostic (H1..H5 print sections) explaining a specific historical amplitude-scale confusion; not a named production capability (no DPSS/GPR/delay/imaging/selective/weighted/rfi-sweep/noise-gen content). Grouped in docs/inpainting-investigation-brief.md alongside characterise_speckle.py/sim_real_gap.py as data-prep investigation scripts.
- invoked by: data_preparation/real/jobs/audit_real.sh
- note: Docstring in code says fake_mask is 'identical to model/data.py' (inlined duplicate).

### `data_preparation/real/characterise_speckle.py`
Decomposes real (or sim clean/mask) waterfalls into smooth + residual via box-filter, computes residual std ratio and freq/time lag-1 autocorrelation and correlation length, to characterise real speckle statistics for mimicking it.
- evidence: docs/inpainting-investigation-brief.md explicitly lists characterise_speckle.py under 'Information-theoretic diagnostics' alongside speckle_probe.py, bias_diag.py, sampler_sweep.py -- all dead-end probes per project facts; decompose smooth/residual split is the superseded decompose-target era machinery.
- invoked by: data_preparation/simulated/jobs/realify_test.sh
- note: Its only caller (realify_test.sh) is itself in the simulated/ tree, outside this file set -- capability is dead-end speckle characterisation regardless.

### `data_preparation/real/cross_baseline_recoverability.py`
Reads a narrow-band cube directly from an MS, then measures cross-baseline R^2 (predict a flagged visibility from k nearest uv-neighbour baselines) vs within-baseline freq-interp R^2 vs mean-fill, plus uv-coherence vs baseline separation, as a probe of the amplitude recoverability ceiling.
- evidence: docs/imaging-investigation-handover.md and docs/methodology-audit.md both call this 'the probe' that 'ran' once (2026-06-24, R2=-0.09 on an uncalibrated bright calibrator) and treat its finding (recoverability ceiling CONFIRMED) as settled; it is a one-off diagnostic that already produced its answer, not a re-run pipeline stage.
- invokes: data_preparation/real/rfi_bands.py (imported, LBAND_PERSISTENT_MHZ unused in body)
- invoked by: data_preparation/real/jobs/cross_baseline.sh
- note: Borderline: the recoverability-ceiling finding it produced does feed the report narrative, but the script itself is a completed probe, not a repeatable eval stage named in final.md's production list.

### `data_preparation/real/inspect_ms.py`
Samples an MS for flag-fraction stats (per-baseline dead-antenna detection, per-timestamp bad-time detection, percentile flag fractions on usable baselines) to sanity-check a new MS before extraction.
- evidence: No job script, doc, or other .py in the repo references inspect_ms.py (grepped repo-wide, zero hits besides the file itself) -- cannot tell if it is a still-used manual pre-extraction sanity check or an abandoned one-off.
- note: Capability (flag-fraction / dead-baseline audit before extraction) is exactly the kind of quick check final.md's 'inspect before extract' workflow would want, but with zero repo references I can't confirm it is actually still run. | Cross-check: zero callers, single commit, one-off MS inspection.

### `data_preparation/real/jobs/audit_real.sh`
SLURM job (Main, 30min) running audit_real.py against a real variant h5 (default v1_upsample512.h5) inside ASTRO-PY3.10.sif.
- evidence: Sole caller of archive-classified audit_real.py; classified by the capability it invokes.
- invokes: data_preparation/real/audit_real.py --data $DATA --n 200

### `data_preparation/real/jobs/cross_baseline.sh`
SLURM job (Main, 4h, 128GB) running cross_baseline_recoverability.py against a named target MS/column/freq-window, writing an .npz result tagged by $TAG.
- evidence: Sole caller of archive-classified cross_baseline_recoverability.py.
- invokes: data_preparation/real/cross_baseline_recoverability.py --ms $MS --column $COLUMN --freq-min $FMIN --freq-max $FMAX --n-chan $NCHAN --out $OUT

### `data_preparation/real/jobs/flag_real_test.sh`
SLURM job (Devel, 1h) doing a quick smoke-test version: CASA split extracts scan 1 into a subset MS, runs tricolour on the subset, then calls extract_ms.py with --max-patches-per-bl 5 --max-time 512 and visualise_real.py with --patches --max-time 512.
- evidence: Calls extract_ms.py with --max-patches-per-bl and --max-time, neither of which exist in the current extract_ms.py argparse, and visualise_real.py with --patches which also does not exist -- both calls would error immediately, proving this test script targets an earlier version of those scripts and was never updated (superseded quick-test).
- invokes: CASA split, tricolour, data_preparation/real/extract_ms.py (stale flags), data_preparation/real/visualisation/visualise_real.py (stale flag)

### `data_preparation/real/jobs/recover_real.sh`
SLURM job (Main, 30min) running recover_real.py against a real variant h5.
- evidence: Sole caller of archive-classified recover_real.py.
- invokes: data_preparation/real/recover_real.py --data $DATA --n 200

### `data_preparation/real/jobs/recoverable_real.sh`
SLURM job (Main, 30min) running data_preparation/simulated/visualisation/decompose_layers.py against real data with sigma=2, to see how much real amplitude is recoverable structure vs irreducible noise via a 2D low-pass split.
- evidence: Invokes decompose_layers.py, the smooth/residual low-pass decomposition tool from the decompose-target era, which project facts state is superseded ('the smooth/decompose target era is SUPERSEDED').
- invokes: data_preparation/simulated/visualisation/decompose_layers.py
- note: decompose_layers.py itself is outside this file set (lives under simulated/visualisation/) so only the job script is classified here, but its capability (decompose-target era analysis) is definitively archive per known project facts.

### `data_preparation/real/jobs/sigma_sweep_real.sh`
SLURM job (Main, 30min) looping SIG in {1.0,1.5,2.0,3.0,4.0} and running decompose_layers.py at each sigma to find the low-pass cutoff maximising recoverable smooth structure while whitening the residual.
- evidence: Same decompose_layers.py / smooth-residual sigma tuning as recoverable_real.sh -- decompose-target era, explicitly superseded per project facts.
- invokes: data_preparation/simulated/visualisation/decompose_layers.py

### `data_preparation/real/jobs/sim_real_gap.sh`
SLURM job (Main, 30min) running sim_real_gap.py comparing sim dataset.h5 runs against the real v1_upsample512.h5 variant.
- evidence: Sole caller of archive-classified sim_real_gap.py ('sim-real-gap probes' named explicitly in final.md archive criteria).
- invokes: data_preparation/real/sim_real_gap.py

### `data_preparation/real/jobs/sweep_compare.sh`
SLURM job (Main, 30min) running visualisation/compare_sweep.py to compare mean spectra/flag-fraction distributions across the sigma-clip/smooth-bins sweep configs against a flagged MS.
- evidence: Directly wired to sweep_configs.json and the sweep output dir, i.e. the 'sweep-extract era' final.md names explicitly as archive.
- invokes: data_preparation/real/visualisation/compare_sweep.py

### `data_preparation/real/jobs/sweep_configs.json`
8 named parameter sets (smooth_bins, sigma_clip, max_flag_frac) for the sigma-clip/smoothing sweep: baseline, tight_sc, loose_sc, no_sc, wide_sm, narrow_sm, tight_all, loose_all.
- evidence: Consumed only by sweep_extract.sh and sweep_compare.sh, both sweep-extract-era job scripts that call extract_ms.py with a --sigma-clip flag the current extract_ms.py does not have -- confirms this config belongs to a retired extraction-parameter sweep.
- invoked by: data_preparation/real/jobs/sweep_extract.sh, data_preparation/real/jobs/sweep_compare.sh

### `data_preparation/real/jobs/sweep_extract.sh`
SLURM array job (0-7, Main, 4h) reading sweep_configs.json by SLURM_ARRAY_TASK_ID and calling extract_ms.py with --smooth-bins/--sigma-clip/--max-flag-frac then visualise_real.py with --patches, one h5+vis pair per config.
- evidence: final.md explicitly names 'sweep-extract era' as archive; additionally its extract_ms.py call uses --sigma-clip and --max-flag-frac (singular) which do not exist in the current extract_ms.py (which has --max-bl-flag-frac/--max-ts-flag-frac and no sigma-clip at all), and its visualise_real.py call uses --patches which does not exist either -- this job is broken against current code, confirming it as a dead, superseded era.
- invokes: data_preparation/real/extract_ms.py (stale flags), data_preparation/real/visualisation/visualise_real.py (stale flag)

### `data_preparation/real/recover_real.py`
Re-derives whether fake-mask holes (the same holes used in phase-2 self-supervised training/eval) contain recoverable structure: computes context->hole R^2 and lag-1 autocorrelation, edge-vs-interior interp/mean-fill error, to check whether the real-data 'ties mean-fill' negative result is a genuine SNR ceiling or a metric bug.
- evidence: This is the diagnostic behind MEMORY project_speckle_sweep_finding.md ('real amplitude in hole is ~86% irreducible white noise; model can't beat mean-fill at any speckle level') -- an explicitly dead-end/superseded finding, and the fake_mask function here is noted in-code as duplicated from model/data.py purely for offline diagnosis.
- invoked by: data_preparation/real/jobs/recover_real.sh

### `data_preparation/real/sim_real_gap.py`
Loads sim clean/mask/phase and real data/flags/phase h5s and compares amplitude distribution (KS/Wasserstein), lag-1 autocorrelation, radial PSD, phase distribution/circular std, and RFI-mask morphology (persistent-band flag fraction, contiguous band-width), producing comparison PNGs.
- evidence: final.md's archive criteria explicitly names 'sim-real-gap probes' as a dead end; this file is literally that probe (filename and content match one-to-one).
- invoked by: data_preparation/real/jobs/sim_real_gap.sh
- note: Produces figure-like PNGs (amp_dist.png, psd.png, phase_dist.png, flag_profile.png, waterfalls.png) which could tempt a 'figure' classification, but the explicit archive-criteria naming of this exact probe type overrides that.

### `data_preparation/real/usable_subset.py`
Reads an MS, computes per-baseline and per-timestamp flag fractions, flags dead baselines (>95% flagged) and bad timestamps, and reports the flag-fraction percentile distribution over usable (non-auto, non-dead) baselines with counts passing various thresholds -- intended to help pick a --max-bl-flag-frac cutoff.
- evidence: No job script, doc, or other file references usable_subset.py anywhere in the repo (grepped repo-wide, zero hits); functionally it looks like a parameter-selection helper for extract_ms.py's --max-bl-flag-frac but there is no wiring proving it is still exercised.
- note: Same ambiguity as inspect_ms.py -- plausible still-useful manual pre-extraction utility, but unverifiable from repo evidence alone. | Cross-check: zero callers; purpose superseded by extract_ms.py --max-bl-flag-frac.

### `data_preparation/real/visualisation/compare_real_vs_sim.py`
Loads a simulated waterfall from a .npy (+ .meta.npy) file and a real MS, compares them via side-by-side waterfall images, amplitude histogram, and mean-spectrum overlay plots.
- evidence: Depends on a --sim-waterfall .npy/.meta.npy pair, an output format not produced by the current h5-based simulated pipeline (dataset.h5); superseded by sim_real_gap.py which does the same comparison against the current h5 format -- itself already archive, making this even older/more superseded.
- note: No job script in this file set invokes compare_real_vs_sim.py; grep found no other repo references either, consistent with it being a leftover from before dataset.h5 existed.

### `data_preparation/real/visualisation/compare_sweep.py`
Loads a base mean spectrum directly from an MS and per-config patch stats (flag fraction, amp mean/std) from each sweep_configs.json entry's h5, producing a mean-spectrum comparison plot, per-config flag-fraction histograms, and a summary table image.
- evidence: Purpose-built for sweep_configs.json / sweep_extract.sh's sigma-clip sweep, the explicitly-named archive 'sweep-extract era'.
- invokes: data_preparation/real/rfi_bands.py
- invoked by: data_preparation/real/jobs/sweep_compare.sh
- note: The mean-spectrum overlay loop currently plots base_spec with alpha=0 for every config (a no-op/dead code path in the plotting loop) -- another sign this script was mid-edit when the sweep era was abandoned.

### `data_preparation/simulated/jobs/decompose_probe.sh`
Builds realify.py smooth-target variants (structure+speckle) then runs model/diagnostics/speckle_probe.py to test x0-vs-eps training against a noisy vs smooth (decompose) target.
- evidence: Directly exercises the smooth-target ('DECOMPOSE') training recipe, which the project facts state is SUPERSEDED and never a target going forward; tag names ('smooth-target-DECOMPOSE') match the dead decompose-target era.
- invokes: data_preparation/simulated/realify.py, model/diagnostics/speckle_probe.py
- note: Superseded by the noise-free-target recipe (project_noise_free_target memory) which final.md now makes the headline target.

### `data_preparation/simulated/jobs/realify_test.sh`
Runs characterise_speckle.py against real data, builds realify.py nospeckle/speckle variants, then runs model/diagnostics/overfit_test.py x0-vs-eps convergence probes on each.
- evidence: Builds and tests realify.py's smooth+speckle decomposition variants (superseded decompose/speckle-era) purely as a convergence probe; not on the simulate->train->...->evaluate production path and not a figure/validation gate.
- invokes: data_preparation/real/characterise_speckle.py, data_preparation/simulated/realify.py, model/diagnostics/overfit_test.py
- note: Part of the same speckle/decompose investigation lineage as speckle_probe.sh and decompose_probe.sh; all three share realify.py as their only 'build' step.

### `data_preparation/simulated/jobs/simulate_test.sh`
Small-scale dry run of the same simulate pipeline (RUN_ID=test) for manual sanity-checking of pipeline changes before a full run.
- evidence: Calls extract_patches_sim.py with an --img-size flag that script's argparse does not define (only --ms/--output/--waterfall-out/--freq-min/--freq-max/--max-bl-flag-frac/--smooth-bins) — it is stale relative to the current post-tiling-refactor extract_patches_sim.py and would error if run; final.md's own archive list names 'smoke tests' explicitly.
- invokes: simms, crystalball, data_preparation/simulated/add_noise.py, data_preparation/simulated/extract_patches_sim.py, data_preparation/simulated/inject_rfi.py, data_preparation/simulated/visualisation/visualise_simulate.py
- note: Discrepancy noted: the --img-size flag mismatch is concrete evidence this script wasn't updated after the native-extract/tile-in-inject refactor (reflected correctly in simulate.sh and reextract.sh).

### `data_preparation/simulated/jobs/speckle_probe.sh`
Builds a speckle_std sweep (0.00/0.06/0.12/0.18) of realify.py variants at fixed smooth-amplitude std, visualises sim-vs-real speckle, then trains speckle_probe.py to plateau on each to test x0/eta0 recovery vs a smooth target.
- evidence: This is the exact speckle sweep named as a dead end in project facts ('speckle probes are dead ends'; memory: real amplitude in-hole is ~86% irreducible white noise, model can't beat mean-fill at any speckle level).
- invokes: data_preparation/simulated/realify.py, data_preparation/simulated/visualisation/vis_speckle.py, model/diagnostics/speckle_probe.py
- note: None.

### `data_preparation/simulated/make_bright_sky.py`
Scales every source flux in sky_model.txt by a constant factor (default 40x) to produce sky_model_bright.txt, a brighter synthetic sky for higher-SNR test runs.
- evidence: final.md explicitly instructs using 'the current normal sky_model.txt range, NOT the old 40x bright one' going forward — this script's only output (the 40x bright sky) is the thing being retired.
- note: Not invoked by any job script in this file set (sky_model_bright.txt itself is checked in and used as a default in jobs/simulate.sh, so the bright sky model artifact persists even though the generator script is a one-off).

### `data_preparation/simulated/realify.py`
Rescales/decomposes clean amplitude patches to real-calibrated mean/std, adds correlated speckle noise, and writes a clean_smooth (interpolate+low-pass) target field alongside a noisy 'clean' field and RFI-corrupted 'corrupted' field, for smooth-vs-noisy target experiments.
- evidence: Its entire purpose is producing the clean_smooth decompose-target and speckle-calibration variants used only by decompose_probe.sh, realify_test.sh, and speckle_probe.sh — all three archived; project facts state the smooth/decompose-target era is superseded.
- invokes: data_preparation/real/rfi_bands.py (LBAND_PERSISTENT_MHZ)
- invoked by: data_preparation/simulated/jobs/decompose_probe.sh, data_preparation/simulated/jobs/realify_test.sh, data_preparation/simulated/jobs/speckle_probe.sh
- note: Classified by capability per instructions: every caller of this script is archive-era and its core output (clean_smooth) is the exact thing final.md calls circular/banned ('Never use the interpolate-then-smooth smooth_component as a training or eval target'). No production caller exists in the repo.

### `data_preparation/simulated/simulate_vis.yml`
Stimela2 recipe defining a 'simulate-meerkat' pipeline (simms create-ms + meqtrees predict) as an alternative to the direct simms/crystalball shell pipeline.
- evidence: Not referenced by any script or job in the repo (grep for simulate_vis/meqtrees/stimela_meqtrees turns up only this file); the production simulate pipeline (jobs/simulate.sh) predicts via crystalball directly, not via this stimela/meqtrees recipe.
- note: Looks like an early prototype of the simulate stage superseded by the shell-script version; orphaned in the current pipeline.

### `data_preparation/simulated/sky_model_bright.txt`
40x-scaled version of sky_model.txt (fluxes 0.12-4.92 Jy), the default SKY_MODEL in jobs/simulate.sh and jobs/simulate_test.sh.
- evidence: final.md names this exact file/scale as the thing to retire ('the old 40x bright one'), even though it is still wired as the current default sky in simulate.sh.
- invoked by: data_preparation/simulated/jobs/simulate.sh (as default SKY_MODEL), data_preparation/simulated/jobs/simulate_test.sh (as default SKY_MODEL)
- note: Discrepancy to flag for the restructure: production job scripts still default to this archived sky model; simulate.sh's SKY_MODEL default needs to change to sky_model.txt (or a config-driven flux scale) per final.md, independent of this data file's own classification.

### `data_preparation/simulated/visualisation/decompose_layers.py`
Splits an amplitude waterfall into a 2D-Gaussian-smoothed 'smooth' component and a 'grain' residual, plus phase, and reports lag-1 frequency autocorrelation for each to argue which components are recoverable vs irreducible noise.
- evidence: Implements the smooth/grain decomposition itself (used to justify the now-superseded decompose-target design); invoked only by real/jobs/recoverable_real.sh and real/jobs/sigma_sweep_real.sh (grep), both outside this production file set and part of the sim-real-gap/recoverability probe lineage, not the current dataset examples figure.
- invoked by: data_preparation/real/jobs/recoverable_real.sh, data_preparation/real/jobs/sigma_sweep_real.sh
- note: docs/phase2-handover.md and docs/inpainting-investigation-brief.md list it alongside speckle_probe.py/compare_inpaint.py as an investigation-era diagnostic, consistent with archive.

### `data_preparation/simulated/visualisation/vis_speckle.py`
Plots sim-speckle vs real-data amplitude waterfalls side by side (plus a zoomed grain crop) to visually compare simulated speckle texture against real MeerKAT grain.
- evidence: Only invoked by jobs/speckle_probe.sh, an archived speckle-probe dead end; expects a 'clean_smooth' field that only realify.py (archive) produces.
- invoked by: data_preparation/simulated/jobs/speckle_probe.sh
- note: None.

### `evaluation/jobs/pair_dataset.sh`
SLURM job wrapper for evaluation/make_paired_dataset.py, pairing a noisy dataset.h5 with a NOISE_SCALE=0 clean dataset.h5 to build the retrofit noise-free-target training set.
- evidence: final.md line 40 explicitly forbids resurrecting this: 'Do not resurrect the retrofit pairing script; build this into extraction' — the noise-free target must be produced natively by the sim extraction pipeline going forward, not by this post-hoc pairing job.
- invokes: evaluation/make_paired_dataset.py

### `evaluation/jobs/research_queue.sh`
One-off batch submission script queuing a specific sweep of fakehole_delay.sh jobs (GPR-ell sweep, band-vs-blob geometry, posterior-sampling vs noise_floor, divisor test, all-tiles CI, a persistent-band-location re-extraction) plus a pointer to the oracle imaging-gate jobs, tied to the '2026-07-03 deep-research verdict' points.
- evidence: Header comment: 'Each block maps to one point of the 2026-07-03 deep-research verdict ... Comment out any block you don't want' — a bespoke one-time investigation queue, superseded by final.md's planned config-driven Master Orchestrator; the underlying evaluation script it calls (fakehole_delay_eval.py / fakehole_delay.sh) remains production even though this particular submission script is a one-off.
- invokes: evaluation/jobs/fakehole_delay.sh, data_preparation/real/jobs/extract_variants.sh, inference/jobs/oracle_level0.sh (referenced, not called), inference/jobs/oracle_phasefix.sh (referenced, not called)

### `evaluation/make_paired_dataset.py`
Retrofits a noise-free training target by matching rows (baseline/time/freq) between a noisy dataset.h5 and a NOISE_SCALE=0 clean dataset.h5, renormalising the clean amplitude onto the noisy row's divisor.
- evidence: final.md line 40: 'Do not resurrect the retrofit pairing script; build this into extraction' — this IS that retrofit script (per its own commit message 'noise-free-target test: pair noisy input with clean target'); the capability is being replaced by native paired-field extraction, not reused.
- invoked by: evaluation/jobs/pair_dataset.sh

### `inference/jobs/inpaint_ms.sh`
SLURM GPU job that runs GPU inference (stage 1) and CPU write-back (stage 2) as one combined job.
- evidence: inpaint_infer.sh's own header states it was 'Split out from inpaint_ms.sh so the GPU job needs modest memory'; no currently-used queue script calls inpaint_ms.sh — they all call the split inpaint_infer.sh + inpaint_writeback.sh instead, and the script self-labels 'UNVALIDATED on real data'.
- invokes: inference/inpaint_infer.py, inference/inpaint_write.py
- note: Superseded combined-stage predecessor; the .py files it wraps are production, but this particular .sh wrapper is dead in favour of the two-stage split — flagged per the 'classify by capability, note the discrepancy' instruction.

### `inference/jobs/master_queue.sh`
Launches two parallel chains on different MSes: GROUP A re-runs the level-0 + phase-fix write-back oracle gate on a sim MS; GROUP B runs the selective inpaint (infer/write/image) on a real MS.
- evidence: The script's own comments label GROUP A as validating write-back mechanics (a correctness gate) and GROUP B as a production selective-inpaint experiment — the two halves belong to different buckets in the classification scheme, so no single label fits the whole file.
- invokes: inference/jobs/oracle_level0.sh, inference/jobs/oracle_phasefix.sh, model/diagnostics/jobs/compare_models_real.sh, inference/jobs/inpaint_infer.sh, inference/jobs/inpaint_writeback.sh, evaluation/image_eval.sh
- note: GROUP B duplicates selective_inpaint_queue.sh almost line-for-line (same env vars and job chain) — likely should be de-duplicated in the restructure, with GROUP A split out into a tests/ gate runner. | Cross-check: GROUP B duplicates selective_inpaint_queue.sh near-verbatim; consolidate to the latter.

### `inference/jobs/oracle_pfix_wsweep.sh`
One-off weight-frac sweep (0.05/0.2/0.5) on the already-written ORACLE_PFIX_DATA column, testing whether down-weighting the (already-correct) oracle fill turns its off-source-RMS win into a fidelity win too.
- evidence: docs/imaging-investigation-handover.md frames this explicitly as testing 'Suspect #4' (full-weight hard substitution) in a specific, now-superseded bug-hunt decision tree; it is not among final.md's named kept validation gates (level-0 oracle, hole_pred_check, representation diagnostics).
- invokes: evaluation/set_holes_weight.py, wsclean (oxkat container), evaluation/compare_images.py
- note: Requires oracle_phasefix.sh to have run first (checks for its FITS outputs). It invokes evaluation/set_holes_weight.py, which IS the production weighted-imaging capability (final.md ablation #3) — classified by capability there; this .sh is a one-off historical use of that capability on oracle (non-model) data, not the production sweep itself (that's downweight_delay_queue.sh). | Same lineage — QUESTION for user.

### `inference/jobs/oracle_phasefix.sh`
SLURM job building a 'phase-fix' oracle (true amplitude, phase reconstructed from native-resolution cos/sin instead of the resized wrapped angle) and imaging it vs clean/flagged, to isolate whether phase-angle-resize corrupts the write-back.
- evidence: docs/imaging-investigation-handover.md labels this 'Suspect #1 (NEW, untested)' in a specific historical investigation; project memory (oracle-localization) shows the investigation moved on from this test to repr_diag.py for finer bisection, and repr_diag.py — not this oracle — is the tool final.md names as the kept 'representation diagnostics' gate.
- invokes: inference/oracle_phasefix.py, inference/inpaint_write.py, evaluation/image_eval.sh
- invoked by: inference/jobs/master_queue.sh
- note: Also referenced as a suggested next step in evaluation/jobs/research_queue.sh, but that's a suggestion in an investigation log, not a standing production/validation call site. | Same as oracle_phasefix.py — QUESTION for user.

### `inference/oracle_phasefix.py`
Builds oracle preds using true clean amplitude plus phase reconstructed from native-resolution cos/sin (testing the phase-angle-resize hypothesis); optional --smooth-amp mode writes the smooth_component(clean) amplitude ceiling.
- evidence: Its purpose is the one-off 'Suspect #1' investigation superseded by repr_diag.py's fuller bisection (per docs/imaging-investigation-handover.md and project memory); its --smooth-amp branch explicitly reaches into the superseded smooth_component/decompose-target era ('the DECOMPOSE-model ceiling').
- invoked by: inference/jobs/oracle_phasefix.sh
- note: Standalone; reuses smooth_component from data.py conceptually but reimplements it locally. | PROVISIONAL pending user: intermediate diagnostic from the imaging bug hunt; final.md keeps level-0 + repr_diag + hole_pred_check, not phasefix. QUESTION for user.

### `model/diagnostics/ceiling_check.py`
Scores a saved inpaint npz (clean/pred/mask) against mean-fill and a uniform_filter-smoothed 'noise floor' baseline, plus a structure-only MAE on the smoothed component.
- evidence: No .sh anywhere in the repo invokes ceiling_check (grep across all *.sh found zero matches); its smoothed-proxy 'noise floor' methodology predates and is superseded by noise_free_fill_check.py's real paired clean/noisy h5 approach.
- note: Pure post-hoc analysis script over a static npz; standalone, no dependency on config.py/data.py.

### `model/diagnostics/compare_inpaint.py`
Samples SIM/FINETUNE/optional-SCRATCH checkpoints on real RFI-flagged tiles and renders a side-by-side amp+phase fill comparison, with an optional speckle-resample step for smooth-target models.
- evidence: jobs/compare_inpaint.sh defaults SIM/FT/SCRATCH to phase1_all_decompose / phase2_decompose checkpoints and documents --resample-speckle as being 'for smooth-target models' — the decompose/smooth-target era is explicitly superseded, and compare_models_real.py is the newer, actively-orchestrated equivalent (v6_native512, full-amp, keep-persist).
- invoked by: model/diagnostics/jobs/compare_inpaint.sh
- note: Functionally superseded by compare_models_real.py, which is wired into the live production queue; this earlier version has no current caller in master_queue.sh/selective_inpaint_queue.sh.

### `model/diagnostics/diagnose_model.py`
Ad-hoc probe of a phase-1 checkpoint: in-mask vs out-of-mask noise-prediction error per channel across a few timesteps, and one-step x0 reconstruction MAE.
- evidence: Not referenced by any .sh in the entire repo (grep found zero matches) — a one-off internal debugging script with no pipeline wiring.
- note: Orphan script; safe to archive.

### `model/diagnostics/info_ceiling.py`
Runs 'E1 information ceiling' (context->hole R^2, edge vs interior recoverability) and 'E5 metric alignment' (hole-only MAE vs whole-image PSNR) analyses on an old-format sim h5 with direct clean/mask fields.
- evidence: No .sh anywhere invokes info_ceiling.py; its whole-image-PSNR-is-inflated critique and context->hole R^2 approach match the dead methodology-audit era (superseded once the project settled on hole-region MAE plus the noise-free-target framing).
- note: Orphan script exploring an old-format h5 (top-level clean/mask, not the current data/flags/phase real-dataset schema).

### `model/diagnostics/jobs/compare_inpaint.sh`
SLURM job wrapping compare_inpaint.py to render a sim/finetune/[scratch] amp+phase fill comparison on real data.
- evidence: Defaults SIM/FT/SCRATCH to phase1_all_decompose / phase2_decompose checkpoints (the superseded decompose era); superseded by jobs/compare_models_real.sh which is wired into the live production queue.
- invokes: model/diagnostics/compare_inpaint.py (--data, --sim-ckpt, --ft-ckpt, --scratch-ckpt, --resample-speckle, --smooth-sigma)
- note: Uses ASTRO-GPU-PyTorch-2026-01-28.sif container, matching current cluster convention.

### `model/diagnostics/jobs/overfit_real.sh`
SLURM job wrapping overfit_real.py; defaults to SYNTHETIC=1 (in-memory fringe data, no real dataset needed).
- evidence: Wraps overfit_real.py, whose TRE-based verdict and 'untested wiring' purpose are superseded (see that file's entry).
- invokes: model/diagnostics/overfit_real.py (--n, --iters, --bs, --lr, --predict, --hole-fill, --eta, --synthetic or --data)
- note: Real-data path default /scratch3/users/$USER/rfi/real/dataset.h5 predates the current variants/*.h5 naming convention used elsewhere (v1_upsample512.h5, v6_native512.h5), another sign of staleness.

### `model/diagnostics/jobs/reversible_inpaint.sh`
SLURM job wrapping reversible_inpaint.py, hardcoded to the phase2_decompose finetune checkpoint with --smooth-target on by default.
- evidence: Both its default checkpoint and its default SMOOTH=--smooth-target flag are decompose/smooth-target era — see reversible_inpaint.py's entry.
- invokes: model/diagnostics/reversible_inpaint.py (--data, --ckpt, --output, --methods, --sigma, --n, --steps, --smooth-target)
- note: No production caller found; consistent with the underlying .py being archive.

### `model/diagnostics/jobs/smoke512.sh`
SLURM job combining smoke512.py's batch/timing sweep with overfit_test.py's learning-gate check for the 512-patch phase-1 model.
- evidence: Named for and dominated by the one-off 512-batch-sizing profiling exercise (smoke512.py, no correctness assertions); the appended overfit_test.py call is a secondary bolt-on.
- invokes: model/diagnostics/smoke512.py (--data, --predict), model/diagnostics/overfit_test.py (--data --n 8 --iters 400 --bs 4 --predict --eta 0.0)
- note: Discrepancy: bundles an archive-classified profiler with a validation-classified gate in one job; the restructure should split these rather than keep the job whole.

### `model/diagnostics/jobs/sweep.sh`
SLURM job running recoverability.py once, then overfit_test.py across seven hand-picked hyperparameter configs (hole-fill zero/mean/center, predict noise/x0, amp-only, alternate lr).
- evidence: A hyperparameter-exploration sweep whose conclusions (hole-fill=mean, predict mode, etc.) are now fixed in config.py — matches the 'old ... sweep-extract era' archive category from project facts.
- invokes: model/diagnostics/recoverability.py (--data, --n 200), model/diagnostics/overfit_test.py (--data, --n, --iters, --bs, plus per-config --lr/--predict/--hole-fill/--amp-only)
- note: Both scripts it wraps are independently classified archive/archive(recoverability)-validation(overfit_test-as-capability); the job itself is the archived hyperparameter-search artifact.

### `model/diagnostics/overfit_real.py`
Phase-2 wiring sanity test: overfits either an in-memory synthetic fringe batch or real extracted baselines, then scores fake-hole MAE and TRE against mean-fill.
- evidence: Scores its verdict partly on TRE, an explicitly dead metric era; memory note 'Phase 2 build' flags this exact recipe as 'Untested on cluster', and it predates the now-working phase2 config exercised successfully elsewhere (e.g. compare_models_real.py).
- invokes: config.phase2, data.RealDataset/fake_mask/build_cond/positional_encoding, diffusion.Diffusion, unet.UNet, metrics.mae/tre
- invoked by: model/diagnostics/jobs/overfit_real.sh
- note: Default job config (SYNTHETIC=1) never touches real data at all — purely a wiring smoke test, and one built around a dead metric.

### `model/diagnostics/recoverability.py`
Classical mean-fill/interp/biharmonic/multi-scale-smooth MAE comparison on an old-format sim h5 (direct clean/mask fields) to gauge whether masked amplitude has recoverable structure.
- evidence: Only ever invoked from jobs/sweep.sh's 'RECOVERABILITY (no training)' preamble, a hyperparameter/config sweep whose decisions are now fixed; project memory (methodology audit) records this crude probe being superseded by proper classical DPSS/CLEAN/GPR baselines built elsewhere.
- invoked by: model/diagnostics/jobs/sweep.sh
- note: Not to be confused with the production DPSS/GPR fill baselines referenced in final.md (those live outside model/diagnostics).

### `model/diagnostics/reversible_inpaint.py`
Decomposes real amplitude into a hole-aware low-pass + high-freq residual (gaussian/median/wavelet), level-matches the model's in-hole fill to the low-pass, and resamples texture to reconstruct a full-amplitude 'reversed' fill.
- evidence: Explicitly built around a decompose/smooth-target checkpoint (--smooth-target flag; jobs/reversible_inpaint.sh defaults CKPT to phase2_decompose finetune and SMOOTH=--smooth-target) — the smooth/decompose-target era is explicitly superseded per project facts.
- invoked by: model/diagnostics/jobs/reversible_inpaint.sh
- note: The low/high decomposition + level-match machinery only makes sense for the decompose-era checkpoints this script targets.

### `model/diagnostics/smoke512.py`
Batch-size/timing profiling sweep for the 512-patch phase-1 model: forward+backward seconds/iter and peak GPU memory at increasing batch sizes, with epoch-time projections.
- evidence: No PASS/FAIL correctness assertions anywhere (pure throughput/memory profiling); only invoked by jobs/smoke512.sh, a one-off exercise to size batch/epoch budget for the 512 upgrade.
- invoked by: model/diagnostics/jobs/smoke512.sh
- note: Distinct from smoke_test.py: this is a performance profiler, not a correctness gate, despite the similar name.

### `model/diagnostics/speckle_probe.py`
Trains a small model live, target=smooth vs target=noisy, to test whether the resulting fill recovers 'smooth structure' versus baselines — the literal speckle probe.
- evidence: Project fact states explicitly: 'speckle probes are dead ends' — this file is exactly that probe (see memory note project_speckle_sweep_finding).
- note: No .sh wrapper found for this file specifically in the given job list or repo-wide grep.

### `model/real/beat_meanfill.py`
Real-data eval: samples the model, computes complex-vis/phase MAE vs mean-fill, and stratifies amplitude error by connected-hole area (four size buckets).
- evidence: Tied to the old per-baseline RealDataset variant files under /scratch3/.../real/variants/*.h5 which final.md says NOT to reuse ('Real dataset is re-extracted fresh... don't reuse the old v6_native512.h5 as-is'), and its hole-size stratification isn't named anywhere in final.md's evaluation suite or ablations list.
- invokes: model/config.py, model/data.py, model/diffusion.py, model/unet.py, model/metrics.py
- invoked by: model/real/jobs/beat_meanfill.sh
- note: Genuinely useful diagnostic logic (model-vs-meanfill by hole size) that could be salvaged into the flag-fraction/RFI-width sweep ablation (final.md #6), but as it stands it's a one-off audit tool from the phase-2-audit era, not a named production component.

### `model/real/finetune.sh`
For each real-data variant, trains BOTH finetune-from-sim and from-scratch to an equal iter budget via train_real.py, then evaluates each with eval_real.py and prints a ranking.
- evidence: final.md: 'always train BOTH configs — finetune seeded from the sim model AND from-scratch (the sim-prior contrast is a report result)' — this script is exactly that head-to-head, using the full-amp (non-decompose) fake-hole recipe final.md keeps.
- invokes: model/train_real.py, model/real/eval_real.py
- note: Uses no --smooth-target flag (unlike finetune_decompose.sh), so it is the full-amp/current-recipe sibling; INIT default points at the now-superseded phase1_all checkpoint naming and will need repointing to the new production sim checkpoint, but the launcher logic itself stays. | CORRECTED by orchestrator: greps TRE columns (dead metric) and trains old v1_upsample512/v4_relaxed512 variants; superseded. Phase-2 capability lives in train_real.py.

### `model/real/finetune_decompose.sh`
Decompose-recipe (--smooth-target) real finetune+scratch comparison seeded from the sim decompose checkpoint, with sigma/EMA/fake-mask-mode knobs tuned for that recipe.
- evidence: Its whole purpose is exercising --smooth-target end to end (SMOOTH default 1, SIGMA/EMA tuned for decompose) — exactly the 'old smooth_component/decompose-target era scripts' the classification criteria names as archive material.
- invokes: model/train_real.py, model/real/eval_real.py
- invoked by: model/run_retrain.sh
- note: Contains a SMOOTH=0 escape hatch to run full-amp instead, but that mode duplicates finetune.sh; the script's default and design intent is the decompose comparison.

### `model/real/jobs/beat_meanfill.sh`
Runs beat_meanfill.py across four specific historical checkpoints (sim phase1_all, fullamp finetune/scratch, decompose finetune) on a fixed v1_upsample512.h5 variant.
- evidence: Hardcodes paths into phase2_decompose_fullamp/ and phase2_decompose/ run directories — a one-off comparison from the fullamp-vs-decompose decision point, using the archived beat_meanfill.py and a variant file final.md says will be re-extracted, not reused.
- invokes: model/real/beat_meanfill.py

### `model/real/jobs/eval_real_fix.sh`
Runs eval_real.py across five specific historical checkpoints (sim, decompose finetune/scratch, fullamp finetune/scratch) on a fixed v1_upsample512.h5 variant to compare recipes.
- evidence: Same pattern as beat_meanfill.sh — a fixed comparison across decompose vs fullamp checkpoints from a resolved audit question, tied to a soon-to-be-re-extracted real dataset variant.
- invokes: model/real/eval_real.py
- note: eval_real.py itself (the thing this job calls) is production; this specific hardcoded comparison launcher is the archived part — a clean example of the 'classify the .py by capability, caller may still be archive' split.

### `model/real/train_real.sh`
Basic single-MODE (finetune|scratch) SLURM launcher for train_real.py against one fixed dataset path /scratch3/users/$USER/rfi/real/dataset.h5.
- evidence: Points at a single non-variant dataset.h5 path that predates the /scratch3/.../real/variants/*.h5 layout used by every later real-data script (finetune.sh, compare_variants.sh, finetune_decompose.sh) — superseded by finetune.sh's parameterized, variant-aware, both-configs-at-once launcher.
- invokes: model/train_real.py
- note: git log shows finetune.sh explicitly replaced an earlier single-variant launcher ('finetune_v1.sh'); train_real.sh looks like the even-earlier prototype from the initial Phase-2 build commit (7fccf1d 'Build Phase 2: real per-baseline extraction + mixed-masking training + TRE').

### `model/run_retrain.sh`
Login-node orchestrator: submits sim full-amp + sim decompose training in parallel, then chains real finetune_decompose to start after the decompose checkpoint (afterok dependency).
- evidence: Its entire raison d'être is wiring the decompose checkpoint (INIT=$RUNS/phase1_all_decompose_80ep/best.pt) into finetune_decompose.sh; both are decompose-era per known project facts ('smooth/decompose target era is SUPERSEDED').
- invokes: model/sim/train_sim.sh, model/sim/train_sim_decompose.sh, model/real/finetune_decompose.sh
- note: It does also (harmlessly) submit train_sim.sh (still-production full-amp sim job), but that's incidental to the script's decompose-chaining purpose — final.md's new 'master orchestrator' (config-driven, state-file, resumable) supersedes this ad-hoc one anyway.


## Working docs (docs/*.md, READMEs)

Status relative to final.md. Superseded docs keep historical value; several carry
load-bearing numbers flagged below — the restructure must not lose these.

### `docs/imaging-investigation-handover.md` — PARTIAL
2026-06-25 handover on the sim MS write-back + imaging test: documents the central puzzle that an oracle write-back (true clean visibility into holes) still lost to flagging on continuum RMS, ranks 5 suspects (phase-angle-resize wrap corruption ranked #1), and prescribes a layered Level-0/Level-1 oracle diagnostic. The puzzle itself was later resolved (phase fixed via atan2(resize(sin),resize(cos)) and Level-0 oracle proven exact per methodology-audit.md/tiling-design-brief.md), so its live 'open puzzle' framing is superseded, but the write-back mechanics description, model inventory, and container/path facts are still accurate and useful.
- must not lose: Real MS path `/idia/projects/astro-cirg/data_for_rfi/1570802018_sdp_l0-J2018_5539-corr.ms` (raw sdp_l0, 30 cols, 2226011 rows, 1301 timestamps, 1711 baselines, good time-runs 35-332 and 410-781); the 3 checkpoint paths/descriptions (phase1_all_old, v5_all512_finetune, v5_all512_scratch); GPU note that V100(gpu-005) is ~1.5x slower than A40/A100, use --constraint=A100|A40; DDIM 200 steps ~0.05 units/s vs 50 steps ~0.2/s; casacore maketabdesc (not maketabledesc) + unique dminfo NAME gotcha; container split (torch only in ASTRO-GPU-PyTorch-2026-01-28.sif, casacore+skimage in ASTRO-PY3.10.sif, wsclean in oxkat-0.41.sif); Singularity blocked on login node (srun/sbatch only); push-to-main classifier-blocked for the agent.

### `docs/inpainting-investigation-brief.md` — SUPERSEDED
A neutral investigation briefing (pre-audit) listing observations vs unverified hypotheses for why real-data inpainting looked poor/smooth, centered on the decompose/smooth-target model results and TRE-based comparison table. Since final.md kills both the smooth/decompose target and TRE as dead, the framing and the results table built around them are superseded, though the underlying data facts (real amplitude stats, persistent RFI bands, sim vs real autocorrelation numbers) remain valid inputs.
- must not lose: Real data stats: v1_upsample512.h5 unflagged amp mean~1.00, std~0.214, p99~1.57, max~20.7; persistent RFI bands ~930-960/1170-1300/1525-1630 MHz; real baselines ~45-50% flagged; sim vs real phase autocorr numbers (real ~0.145 vs sim 0.4-0.99); the ruled-out hypothesis list (EMA bug, leaky conditioning, etc.) so they aren't re-litigated; checkpoint/container/GPU-constraint info in section 8.

### `docs/methodology-audit-actions.md` — CURRENT
Synthesizes two independent audits into the verdict that delay/power-spectrum is the correct headline arena (not continuum RMS), diffusion must be benchmarked against DPSS/CLEAN/GPR rather than assumed superior, and the smooth/decompose target was correctly abandoned in favor of full-amplitude. This is the direct methodological predecessor of final.md's eval-arena and baseline choices (delay + continuum arenas, DPSS/GPR baselines) — highly current, though final.md further pivots the training target from full-amplitude to the noise-free clean target, updating item 3.
- must not lose: The DONE list (native tiling + feathered write-back, wrap-safe phase via atan2, delay_spectrum.py metric, DPSS classical baseline in classical_fill.py/delay_spectrum.py --dpss/image_eval.sh DPSS=1); the full citation list (Pagano 2023 arXiv:2210.14927, Chen & Kennedy 2024 arXiv:2411.10529, Kern & Liu 2021, Kennedy & Bull 2022 arXiv:2211.05088, Chakraborty 2022 arXiv:2203.04994, Prasad & Chengalur 2018, Elahi & Bharadwaj 2025, Massoud 2024, RFI-DRUnet arXiv:2402.13867, Saharia 2022); the smooth-fill-oracle-LOST-to-flagging numbers (RMSE 2.85e-4 vs flagged 2.28e-4) vs full-fill-oracle-WINS (1.71e-4); note to fix GOAL.md ref [12].

### `docs/methodology-audit.md` — CURRENT
The raw 2026-06-25 adversarial audit (source document behind methodology-audit-actions.md) with per-question verdicts on arena, tool choice, tiling, and architecture, plus explicit corrections to earlier framing (killing the 'U-Paint 10^4 catastrophe' claim). Still current as the primary citation source and the original reasoning trail; methodology-audit-actions.md is the condensed/updated version so treat this as backing detail.
- must not lose: The per-question verdict table (Q1-Q8); the explicit correction that Pagano's CNN is presented as viable, not catastrophic; the note that GOAL.md ref [12] (arXiv:2604.01531) is a bogus future-dated ID needing a fix; the action-list priority order.

### `docs/ms-writeback-plan.md` — PARTIAL
Design doc for the inverse-extraction MS write-back (inference/inpaint_ms.py): documents the forward transform to invert (channel crop, pol reduction, divisive norm, resize), the per-unit inverse steps, decisions (write only hole pixels, broadcast one V to all pols, new INPAINTED_DATA column, optional --unflag), and required safety assertions plus the mandatory sim round-trip validation before running on real data. Core mechanics remain valid and are exactly what final.md's write-back stage still needs, but it predates tiling (assumes one chan_lo/native_n_chan per baseline, not per-unit freq_lo) and predates the noise-free clean target and per-arena noise_floor split — needs updating, not replacing.
- must not lose: The safety assertions (time-major row check `row=(time_lo+t)*n_baseline+baseline_id`, native_n_chan==chan_hi-chan_lo, row-index bounds); the decision to always write a new column never overwrite DATA; the required sim-round-trip validation gate (complex-vis MAE in holes, known-pixels-unchanged regression check) before any real-data write-back is trusted.

### `docs/phase1-science-log.md` — CURRENT
Detailed, honest science log of Phase 1 sim training: the critical loss-leak bug and fix (Palette contract x_in=keep*x0+mask*xt), the DDIM eta=0 sampler decision backed by bias/sampler-sweep diagnostics (unbiased fill, texture ratio ~0.748 not 0.27), the amplitude-generalization failure traced to white-noise masked content (R^2~0) on realistic sim, and the bright-sky control (run2) that validated the pipeline (complex MAE 0.058). This remains a valid, citable historical record for the report even though the specific run2/phase1_all numbers will be superseded by the new 10-run + noise-free-target training.
- must not lose: The Palette train/sample contract bugs and fixes (loss leak, frozen-chain re-noising bug); the DDIM eta=0/no-RePaint-benefit finding with numbers (x0_pred MAE ~0.12 vs stochastic DDPM ~0.29 vs mean-fill ~0.26); the bias diagnostic conclusion (signed bias -0.0038, flat across depth/width/structure = no fixable systematic error); the WIDE-band-is-MOST-accurate finding (visualisation artifact, not a real defect).

### `docs/phase2-handover.md` — SUPERSEDED
2026-06-22 handover concluding the decompose/smooth-target finetune is 'the best real model' (beats interp, TRE 40% better than mean-fill) and full-amplitude is degenerate on real data, with checkpoint inventory and the EMA-freeze bug writeup. Since final.md kills the smooth/decompose target and TRE entirely as dead, this doc's headline recommendation is superseded, but the measured facts (irreducible noise fraction, EMA landmine, checkpoint locations) remain valid inputs to the restructure.
- must not lose: The EMA-decay-0.9999-froze-eval-at-sim-init bug and its fix (auto-scale EMA to run length / --ema-decay, job uses 0.999) — explicitly called out as a landmine to keep guarding against in final.md; checkpoint paths for phase1_all, phase1_all_decompose(_80ep), phase2_decompose/v1_upsample512_{finetune,scratch}, phase2_decompose_fullamp; the ~67% irreducible white-noise-in-real-amplitude measurement; GPU pinning notes (gpu-006 A40, qos-interactive for quick jobs).

### `docs/refactor-plan.md` — SUPERSEDED
2026-06-23 refactor plan reprioritizing the metric stack and sampling defaults around noise-dominated data (demote amplitude MAE, add noise_floor_ratio and FWD, standardize eta=0/noise_floor=auto sampling, flagged an L2-loss experiment and the then-unbuilt MS write-back). Most of §1 and §2 were completed (repo cleanup to archive/, noise_floor_ratio wired into train.py/train_real.py, eval_real.py/inpaint_viz.py/compare_inpaint.py noise-floor flags) and the plan's remaining open items (FWD metric, L2 experiment, mean-fill tightening, write-back) were superseded/absorbed by the later, larger methodology audit and now by final.md's restructure — kept in detail below since it's the direct predecessor of the current restructure.
- must not lose: See detailed refactor_plan_summary field.

### `docs/research-noise-dominated-inpainting.md` — PARTIAL
2026-06-19 deep-research synthesis (25 sources, adversarially checked) establishing that masked amplitude noise cannot be inpainted (information-theoretic, Blau & Michaeli perception-distortion theorem), and recommending decompose-then-inpaint plus a pivot to image/power-spectrum domain evaluation with phase as the headline contribution. The domain-pivot insight (don't judge on per-pixel amplitude MAE; judge in image/delay space) is foundational and CURRENT — it is exactly what final.md's delay+continuum arena choice builds on — but the specific 'decompose-then-inpaint, resample noise at inference' recommendation is superseded now that smooth/decompose is dead and the noise-free clean target is the new headline.
- must not lose: The Blau & Michaeli perception-distortion citation (arXiv:1711.06077) and the reasoning that 'beats mean-fill on MAE' is not achievable/expected for noise-dominated holes; the Pagano 2023 (arXiv:2210.14927) citation for HERA-style inpainting judged in delay/PS space; the FWD (arXiv:2312.15289) and PMRF (arXiv:2410.00418) references for distributional/no-ground-truth metrics; the explicit refutation to NOT cite Cohen et al. ICLR2024 arXiv:2310.16047 as 'posterior collapses to one completion' (it says the opposite).

### `docs/sampling-investigation.md` — CURRENT
Research note establishing that stochastic DDPM/RePaint sampling degrades numerical fidelity (injects noise, MAE worse than mean-fill) purely because of the perception-distortion tradeoff, and that deterministic DDIM (eta=0) is the correct choice for point-estimate fidelity, with noise_floor='auto' as a decoupled post-hoc step for statistical/texture realism. This finding and its two-stage (DDIM signal + post-hoc noise) approach is exactly what final.md's per-arena write-back logic (continuum=noise_floor none, delay=matched-grain) is built on — fully current.
- must not lose: The numeric table (interp ~0.10, model x0_pred ~0.12, mean-fill ~0.26, stochastic DDPM ~0.29); the RePaint re-noising-the-hole-each-step failure mode (froze the chain, produced constant garbage) as a documented dead end not to retry; the decision that DDIM eta=0 + post-hoc noise_floor='auto' decouples signal recovery from texture realism.

### `docs/tiling-design-brief.md` — PARTIAL
Design brief for native-frequency-resolution tiling (2 tiles of 512 native channels covering the 898-channel band, ~25% overlap, feather-blended at write-back, per-tile positional encoding) plus the methodology corrections (delay-space headline, DPSS/CLEAN/GPR baselines) later folded into methodology-audit(-actions).md. The tiling geometry and write-back/PE-per-tile design are directly reused by final.md ('frequency tiles keep their deliberate overlap... do not fix that'), so mechanics are CURRENT; however its own 'TARGET DECIDED: full-amplitude + delay-space fabrication guard' is now superseded by final.md's noise-free clean target as the headline.
- must not lose: The exact tiling geometry (ceil(898/512)=2 tiles, tile0=[0:512] tile1=[386:898] native ch, ownership split at ch 449, ~126ch/25% overlap, feather-blend only affects HOLE pixels); the schema fields to add (freq_lo, native_n_chan, freq_min_patch/freq_max_patch per tile) — noting extract_variants.py (real) already had this schema, reused for sim; the write-back file list (inpaint_write.py, set_holes_flag.py/set_holes_weight.py, oracle_level0.py/repr_diag.py/oracle_phasefix.py) needing the per-unit freq_lo offset instead of global chan_lo; the required pre-retrain validation gate (repr_diag freq-resize error -> ~0, tiled-vs-non-tiled oracle delay-space ablation must match high-delay power); anchor numbers (clean 1.262e-4, flagged 2.461e-4, oracle 3.107e-4, phase-fix-oracle 2.192e-4 RMS/2.562e-4 RMSE/7% peak overshoot); real MS path and container/GPU notes repeated from the handover doc.

### `README.md` — SUPERSEDED
Root README — a short stub describing the two-phase goal, TRE as the real-data metric, and claiming GPU training uses `ASTRO-GPU.simg` (contradicted elsewhere: that container has no PyTorch — training uses ASTRO-GPU-PyTorch-2026-01-28.sif per model/README.md and multiple docs) and says the pipeline is 'being built from scratch'. Explicitly targeted for a full rewrite by final.md (quickstart, results summary, repo layout, ilifu assumption, regeneration instructions, license/citation).
- must not lose: Nothing load-bearing here beyond the general two-phase framing; the container claim is actually wrong and should NOT be carried forward as-is into the new README.

### `model/README.md` — SUPERSEDED
Describes the model's I/O contract (3-channel amplitude+cos+sin, 11 input channels, PE, hole_fill/build_cond conditioning), training/eval commands, and GPU/container gotchas (P100 nodes unusable, V100 batch=16 vs A40/A100 batch=32, ASTRO-GPU-PyTorch-2026-01-28.sif required). It describes 256x256 patches and TRE as a phase-2 stub — both now stale (production data is 512x512 native-tiled per tiling-design-brief, and TRE is dead per final.md). The channel/PE/conditioning contract description itself remains structurally accurate and is a good template for the restructured docs.
- must not lose: The channel contract (3-channel amp+cos+sin, avoid raw-angle wrap by using cos/sin, in_channels=11 = noisy x_t(3)+cond(3)+mask(1)+PE(4)); the true-inpainting-not-RFI-removal explanation of build_cond zeroing masked conditioning pixels (a previously-fixed leak, do not reintroduce); GPU/container facts (P100 gpu-001-004 unusable sm_60, V100 batch16 vs A40/A100 batch32, ASTRO-GPU.simg has no torch); dataset split convention (90/5/5 train/val/test, fixed split_seed, MAX_PATCHES caps only train).

### `data_preparation/simulated/README.md` — SUPERSEDED
Documents the dataset.h5 schema produced by extract_patches_sim.py + inject_rfi.py at 256x256 patch size, listing all datasets/attrs and explaining why the write-back metadata fields exist, plus the known pol-reduction approximation (amp=mean(|V|), phase=angle(mean(V)) are not a matched pair). Patch size (256, not 512), missing tiling fields (freq_lo/native_n_chan), and absence of the noise-free clean-phase-target fields (`phase_target`) mean this schema doc is now stale and needs a rewrite for the restructure, though its explanatory structure is a good template.
- must not lose: The pol-reduction caveat (amp=mean(|V|) vs phase=angle(mean(V)) is not a matched pair — documented as a known approximation, not a bug) — must carry forward into any new schema doc; the full field-by-field schema table structure as a template for documenting the new (tiled, noise-free-target) schema.

### `archive/README.md` — PARTIAL
Maps diagnostic scripts archived on 2026-06-23 (bias_diag, infer_compare, gen_sweep, sampler_sweep, viz_eta, extract_windows, merge_windows) to their replacements. Accurate for that snapshot, but multiple later docs (methodology-audit-actions, refactor-plan) reference further scripts as superseded/to-archive (e.g. old TRE-era metrics, speckle probes) that are not yet reflected here — final.md explicitly plans more archiving during the restructure, so this doc is a partial/incomplete map that will need extending, not a wrong one.
- must not lose: The existing mapping table (script -> replacement -> note) as the pattern to extend; the note that archived jobs/*.sh had their internal script paths repointed to archive/diagnostics/ so they remain independently runnable.
