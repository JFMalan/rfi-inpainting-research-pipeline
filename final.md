# Final restructure — publishable pipeline

Pivot this project to its final publishable version. We are working on a dedicated restructure branch (created off `research`); it merges to `main` when done. The end state: a clean, reproducible pipeline someone who has never seen this repo can understand, run end-to-end on ilifu, and use to regenerate every result and figure in the research report.

Use ultracode for this — it is a large job and needs to be done correctly.

## Ground rules (read before touching anything)

1. **Reuse, don't rewrite.** Every capability needed below already exists in this repo and is cluster-tested. This job is restructuring: move, rename, unify configs, delete dead paths. Do NOT re-implement working logic — we cannot afford to re-debug the pipeline. The only sanctioned new code is explicitly listed in "New code allowed" below.
2. **Verify before changing.** These instructions may contain mistakes. If something here contradicts what the code/results actually show, or seems wrong, verify (code, git history, `.claude/docs/`) and ask me rather than silently obeying or silently deviating.
3. **Nothing is lost.** Tag the pre-restructure commit (e.g. `pre-restructure`) before starting. Move superseded scripts to `archive/` rather than deleting. Analyze everything the project currently does first, so no functionality on the production path is dropped by accident.
4. **Cluster reality.** All runs happen on ilifu via SLURM. Never guess ilifu specifics — verify against `.claude/docs/`. We repeatedly hit OOM and wall-time kills: overshoot resources rather than undershoot (training walls 24h+, write-back 128GB, imaging 64GB/4h are known-good numbers).
5. **Code style** per `.claude/CLAUDE.md` (human-researcher style, no AI-sounding comments). Every cluster script logs: device actually in use, setup milestones, periodic progress with rate, all flushed.

## Keep vs archive

**Keep (the production path):** anything that (a) is on the pipeline simulate → train → extract real → finetune → inpaint → write-back → evaluate, (b) regenerates a report figure or table, or (c) is a validation gate proving pipeline correctness — the level-0 oracle write-back test, `hole_pred_check`, the representation diagnostics. Validation gates move to a `tests/` (or `validation/`) area, not archive.

**Archive:** one-off diagnostics, superseded experiments, dead ends (old TRE metric era, speckle probes, smoke tests, etc.). When in doubt, archive and note it.

## Target architecture

- **Config-driven, YAML.** Two levels:
  - `configs/telescope/<name>.yaml` — instrument definition: band edges, channel count/width, dump time, SEFD profile (freq nodes + Jy), antenna layout / simms model, persistent-RFI band list, typical source flux scale.
  - `configs/experiment/<name>.yaml` — run parameters: dataset counts, noise scales, epochs, batch, sampling settings, paths.
- **Telescope scope:** visibility-based interferometers only. MeerKAT L-band is the validated instance. Ship a HERA config as a documented stub (correct band, SEFD, layout, uvh5 noted as the ingest format) with a README note that it is untested — the point is that the pipeline reads instrument parameters from config, not that HERA is production-supported. MeerKLASS is dropped (single-dish intensity mapping — different data type, out of scope).
- Flexibility comes from the config files; do NOT add speculative abstraction layers (YAGNI).

## Simulated data (phase 1)

Produce **10 training runs + 1 held-out test run**, fully deterministic (fixed seeds; the datasets live on `/scratch3` which auto-deletes after 90 days, so *regenerability from scripts is the storage story* — this must work from a single command).

Per run:
- **Sky:** random sky per run (existing `make_random_sky.py`), fluxes matched to the telescope config's realistic scale (the current normal `sky_model.txt` range, NOT the old 40x bright one). Vastly different skies across runs — this is the main diversity axis.
- **Thermal noise:** uniform draw per run in **0.7–1.0x telescope SEFD** (test run at exactly 1.0x). Noise diversity is a minor axis — our own results show the model generalizes 1x→4x at inference — so don't over-invest here; sky and RFI diversity matter more.
- **RFI injection:** as diverse as possible across runs (band widths, blob density, persistent-band mix), mean flag fraction per run ≤50%.
- **Time windows:** increase `SYNTHESIS` so each baseline yields **≥2 non-overlapping 512-bin time windows** (~2.4h at 8s dumps). Frequency tiles keep their deliberate overlap (feathered blending needs it) — do not "fix" that.
- **Sample budget:** total training samples < 100,000. Arithmetic: 10 runs x 2016 baselines x 2 time windows x 2 freq tiles ≈ 80,640 — fits.

**Training target (critical, this is our headline finding):** the target is the **noise-free complex signal — amplitude AND phase** — while the input/conditioning stays the noisy observation. The sim pipeline must generate the paired clean/noisy data natively: snapshot the pre-noise DATA column, extract BOTH clean amplitude and clean phase as target fields alongside the noisy input fields, with the divisive-norm divisor computed from the noisy data and shared. Do not resurrect the retrofit pairing script; build this into extraction. (Currently only amplitude is clean in the paired set — the phase target must become clean too, which needs a separate `phase_target` field so the noisy phase still feeds the conditioning.) Never use the interpolate-then-smooth `smooth_component` as a training or eval target — it is circular (the model gets scored against interpolation).

## Training

- **Phase 1 (sim):** existing `train.py` on the 10 runs. Val metrics already include the mean-fill baseline (`amp_mf`, `beats_mf`) — keep that. Make the val-eval sampling step count configurable (200-step eval was costing ~5h per run inside training walls).
- **Phase 2 (real MeerKAT):** always train BOTH configs — finetune seeded from the sim model AND from-scratch (the sim-prior contrast is a report result). Cap at 40,000 real samples, excluding samples >85% flagged, split by baseline as now. Real-data target: keep the existing fake-hole self-supervised recipe (noisy target) unchanged — there is no clean truth on real data; amplitude recovery leans on the sim prior, and the report documents this limitation explicitly (our sim results show a noisy target caps in-hole amplitude at mean-fill).
- **Real dataset is re-extracted fresh** from the source MS under the new config-driven pipeline (don't reuse the old v6_native512.h5 as-is), so the report's real dataset is itself reproducible from the restructured code. Then re-run the finetune/scratch pair on it.
- After phase 2, always produce test-sample panels: some tiles inpainted fully, and some inpainted selectively (only non-persistent-band RFI filled, persistent bands left flagged) — the keep-persist machinery exists.

## Evaluation suite (the report's results)

**The goal: beat flagging in the delay spectrum, and in the continuum image where possible.** Flagging is the baseline-to-beat everywhere.

Variants compared in BOTH arenas (continuum wsclean image on sim with clean truth; delay spectrum):
1. **Flagged** (standard practice — the target)
2. **DPSS** classical fill
3. **GPR** classical fill (constant-mean version — the zero-mean one reverts to prior in wide gaps)
4. **Inpainted — everything** (all RFI filled)
5. **Inpainted — selective** (non-persistent bands filled, persistent bands stay flagged)

**Per-arena write-back (important, use the right one automatically):** continuum imaging uses the smooth fill (`noise_floor=none` — grain only injects noise into the coherent sum); delay-spectrum analysis uses the matched-grain fill (noise resampled at the locally estimated sigma so the hole texture matches its surroundings and doesn't leak). The eval scripts should pick the correct mode per arena without manual switching.

**Metrics:** continuum RMSE-vs-clean + dynamic range (sim); wlogP-RMSE + hi-delay ratio (delay). **Every headline "X beats Y" claim carries a bootstrap CI** — the delay eval already does this; extend the pattern to the continuum comparisons.

## Ablations (all reuse existing machinery — restructure, don't rebuild)

1. **Massoud et al. component ladder (the important one).** Their data/code isn't available; the comparison is their *method reimplemented on our benchmark*. Baseline rung R0 = the Massoud recipe adapted minimally to our 512x512 tiles: generic U-Net DDPM, amplitude-only channel, L1+L2 loss, their frequency positional encoding + mixed masking (those are their contributions and stay in R0), noisy target, PSNR/MSE-on-masked-region eval. Then add our techniques one rung at a time, retraining each rung:
   - R1 = R0 + divisive normalisation
   - R2 = R1 + cos/sin phase channels (complex reconstruction)
   - R3 = R2 + noise-free target
   - R4 = R3 + sampling/write-back techniques (noise_floor, matched grain) — inference-only, no retrain
   Compute budget: every rung trains on the SAME fixed subset of 2–3 runs (~16–24k samples) for ~30 epochs (~20–25h each) — identical data and epoch budget across rungs so the comparison is fair; only the final production model trains on the full 10-run set. Score every rung on the same held-out test run with PSNR/MSE (their metric) AND our metrics (complex MAE, delay, continuum). Frame honestly in the report: Massoud was simulated-only, 191 samples, no imaging/delay evaluation — this is method-vs-method on our benchmark, not results-vs-their-paper.
2. **Clean vs noisy target** — already demonstrated (amp MAE 0.208 vs 0.340 at 1.0x SEFD; continuum + delay imaging wins); productionize as a first-class ablation on the final datasets. This is a headline result.
3. **Weighted imaging** — the visibility-weight downweighting of filled pixels at imaging time (existing `set_holes_weight` / `WEIGHT_FRAC`): sweep {0, 0.2, 0.5, 1.0} on sim and real, continuum metric.
4. **Sampling techniques** — existing `fakehole_delay_eval` variants: `noise_floor` {none, 0.3, 0.5, matched}, DDIM steps {25, 50, 100, 200}, eta, RePaint resampling (`repaint_u`), posterior ensemble.
5. **Native tiling vs downsample-512** — existing real-data variants comparison.
6. **Flag-fraction / RFI-width sweep** — existing `rfi_width_sweep` machinery with the final model. This scopes the continuum claim: we know the continuum win is regime-dependent (it holds at ~37% flag fraction on faint fields robustly to context noise; flagging won on the old bright-sky test) — the sweep maps the crossover.
7. **Noise generalisation** — existing lecturer-test machinery: final model evaluated at 0x/2x/4x SEFD without retraining (and note the noise-free 0x anomaly honestly if it persists).

## Figures pipeline

A `figures/` area with one script per report figure, regenerating it from checkpoints/metrics on disk (no one-off hand-edited plots). Figures to cover at minimum: dataset examples per run (input / RFI mask / clean target); training curves (loss + val metrics vs epoch, with the mean-fill line); the noise-threshold/recoverability curve; fill-check panels (observed / target / smooth fill / matched-grain fill / delay spectra); continuum image comparisons with residuals; delay-spectrum comparisons; the Massoud ladder chart; the flag-fraction crossover plot; selective-inpaint panels (fill / kept-flagged overlays). Each stage of the pipeline should have at least one image proving it works.

## Master orchestrator

Not a monolithic script — a **SLURM submission orchestrator**: one entry point (e.g. `run_pipeline.sh <experiment.yaml>`) that submits every stage with `afterok` dependency chains across the right partitions, writes a state file (stage → job id → status), and supports **resume**: rerunning after a failure skips completed stages. Per-stage logs in one predictable place; a status command that shows the chain. Fail loudly with a clear message about which stage died and where its log is. Stage-level entry points remain individually runnable (we always debug stage-by-stage).

## Reproducibility & housekeeping

- Fixed seeds end-to-end; documented in the experiment config.
- Canonical checkpoints live on `/idia` (not scratch); keep a short model inventory (name → what it is → path) in the README or a models doc.
- Containers: document exactly which image each stage uses. Known landmine: `ASTRO-GPU.simg` has NO PyTorch — training/inference use `ASTRO-GPU-PyTorch-2026-01-28.sif`; casacore/skimage in `ASTRO-PY3.10.sif`; wsclean in `oxkat-0.41.sif`.
- README rewritten to be genuinely useful: what this is, results summary, repo layout, quickstart (configs → one command), the ilifu/SLURM assumption stated up front, how to regenerate datasets/figures, license + citation entry.
- Update `.claude/CLAUDE.md` and `GOAL.md` for the new layout; keep `.claude/docs/` as-is.
- Keep the validation gates runnable and documented (oracle level-0 write-back == clean; `hole_pred_check`; repr diagnostics).

## New code allowed (only this)

- GPR MS write-back module analogous to `dpss_fill_write.py` (GPR fill exists for the fakehole eval; the imaging arena needs it written into an MS column).
- Clean-phase target plumbing (`phase_target` field through extraction, dataset, and loss).
- The YAML config layer, orchestrator/state file, and figures scripts.
- Massoud R0 baseline config (should be mostly config/flags on the existing trainer, not a new trainer).

## Known landmines (do not rediscover these the hard way)

- `noise_scale=0`: extraction prefers `CORRECTED_DATA`, which is empty unless `add_noise` copies DATA across (fix already in `add_noise.py`) — keep that behaviour in any refactor.
- `smooth_component` interpolates across holes — as a target it makes the metric circular. Never a target.
- EMA 0.9999 once froze eval at init (phase-2 audit) — keep the EMA verify behaviour.
- Write-back: `RESET_COL` is required when a column is reused across experiments (stale-fill contamination); feathered blending is negligible for continuum but keep it for tile seams.
- GPR must be the constant-mean variant.
- The divisive-norm divisor is a 64-bin running mean of the noisy data; any clean/noisy pairing must renormalize onto the shared (noisy) divisor.
- Val-eval at 200 sampling steps inside training costs hours — keep it configurable.
- Scratch auto-deletes at 90 days; anything on scratch must be regenerable by script.

## Process

Before the big restructure, run whatever small verification tests are genuinely necessary (not more). Research anything I've missed that the report will need and ask me about it. Ask me anything unresolved — especially where these instructions might be wrong — rather than assuming. At the end I want: clean repo, one orchestrator, YAML configs, figures pipeline, useful README, and every report result regenerable from scratch.
