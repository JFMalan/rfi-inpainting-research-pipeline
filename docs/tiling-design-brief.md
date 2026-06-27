# Tiling design brief: native-frequency-resolution inpainting

Handoff for the agent implementing native-frequency-resolution **tiling** in the RFI-inpainting
pipeline. Read this in full, plus `docs/imaging-investigation-handover.md`, `docs/methodology-audit.md`,
and the auto-memory notes `project_oracle_localization.md` / `project_resolution_research.md` /
`project_sim_dataset_inventory.md`, **before proposing anything**. Present the design (end of this doc)
for sign-off **before** writing code.

## READ FIRST — methodology corrections (from the adversarial audit; supersede the rest of this doc)
The audit (`docs/methodology-audit.md`) shifts the methodology around the tiling work. Honour these:
- **Headline metric = delay/power-spectrum, NOT continuum RMS.** All peer-reviewed RFI-gap precedent
  judges inpainting in delay space; continuum already tolerates gaps via cross-baseline uv redundancy.
  Demote continuum off-source RMS/DR to a "does-no-harm" sanity check.
- **Benchmark against classical baselines (DPSS, CLEAN, +GPR if time), not just flagging/mean-fill.** The
  diffusion model is the challenger; frame it as "does a learned prior recover the ~⅓ coherent structure
  better than linear filters, at the cost of statistical tractability?" Don't presume superiority.
- **Tiling validation MUST be a delay-space oracle ±tiling ablation** (seams inject high-delay power). The
  tiled oracle's high-delay power must match the non-tiled oracle's. Load-bearing; do not assume feathering
  is safe without it.
- **Recoverable/irreducible split is a FINDING, not a failure.** Wide fully-flagged persistent bands are
  largely irreducible (white noise is incoherent — Pagano 2023; contiguous-vs-random — Chakraborty 2022);
  stop optimizing against them.
- **TARGET DECISION PENDING (ask the user):** the audit recommends switching from the smooth/decompose
  target to a **full-amplitude target + delay-space fabrication guard** (over-smoothing causes signal loss
  — the ~6% peak suppression). This REVERSES the "keep smooth target" note later in this doc. Confirm with
  the user before choosing `train_sim.sh` (full-amp) vs `train_sim_decompose.sh` (smooth).
- The earlier "U-Paint was catastrophic (10^4)" framing is an OVERREACH — Pagano presents the CNN as
  viable/transferable. Don't lean on it.
- Fix `GOAL.md` ref [12] (Luo et al. arXiv:2604.01531 is a future-dated, bogus ID).

## Where we are (verified)
- Goal: inpaint RFI gaps in visibilities, write back to an MS, beat plain flagging on the sky image.
- Model: a 512×512 conditional diffusion U-Net; 3 channels = amplitude, cos(phase), sin(phase).
- Write-back MECHANICS are proven correct: a Level-0 native-passthrough oracle
  (`inference/oracle_level0.py`) reproduced the clean image exactly (RMSE 3e-7); a per-unit row-map
  verifier passed 2016/2016.
- The loss was in the h5 REPRESENTATION. Two parts, both measured by `inference/repr_diag.py` on sim run1:
  1. **Phase** was stored as `resize(raw angle)`, corrupting at 2π wraps (20.7° RMS). **FIXED** in code:
     all 3 extractors now store `atan2(resize(sin), resize(cos))` (commit e206329, helper `resize_phase`).
     Validated: oracle off-source RMS 3.107e-4 → 2.192e-4 (now beats flagged 2.461e-4 on RMS).
  2. **Amplitude 512-resize round-trip (~12% error)** — the REMAINING problem, your target. Confirmed
     it's the FREQUENCY axis: native 898 ch → 512 is a 1.75× undersample (the big loss); time 540→512
     is ~1.05× (small). Local test: full 512² rt error 0.077; freq-only 0.070; time-only 0.045.
- Phase-fix oracle imaging (ceiling for a perfect model): beats flagging on off-source RMS but LOSES on
  fidelity-to-clean (RMSE 2.562e-4 vs flagged 2.165e-4) and overshoots the true peak by 7%. The amp
  512-undersample is what's left. (Divisive-norm recombination was tested and is NOT the cause.)

## Research verdict (deep-research, 18 sources, on-domain)
- Tile seams / hard gap edges are dangerous for interferometry: discontinuities Fourier-ring into
  power-spectrum + image artifacts (Pagano 2023 arXiv:2210.14927; Chen 2024 arXiv:2411.10529). HERA
  pre-fills gaps with CLEAN to avoid this.
- Learned inpainters can fabricate fine-frequency structure (Pagano's CNN edged out by classical on fine
  structure as noise drops — NOT the "10^4 catastrophe" earlier framing claimed). Guard against fabrication
  by MEASURING it in delay space, not necessarily by forcing a smooth target (see TARGET DECISION above).
- Inpainting only reliably reproduces SMOOTH/foreground modes; fine stochastic structure is irreducible
  (matches our 67%-grain finding). Its proven win is the power-spectrum/DELAY domain, maybe not continuum.
- Decision: FREQUENCY TILING with overlap, REUSE the 512 model (NOT a 1024 model, NOT cascaded
  diffusion). ADD a delay/power-spectrum metric as the HEADLINE and classical baselines (DPSS/CLEAN/GPR).

## Proposed tiling scheme (verify with the user)
Per baseline, native is 540 time × 898 freq (band 900–1650 MHz; MS attr `chan_lo=24`, `full_n_chan=898`).
- **Frequency:** tile into `ceil(898/512)=2` tiles, each exactly 512 native channels wide → upsampled to
  512 = identity (no freq undersampling). Tile 0 spans native ch [0:512], tile 1 spans [386:898]; overlap
  = ch [386:512] (~126 ch, ~25%).
- **Time:** keep one tile, 540→512 (1.05×, negligible). (Open Q: tile time too?)
- **Ownership + blend:** partition ownership at the band midpoint (tile0 owns [0:449], tile1 owns
  [449:898]), both interior to their 512-wide span. For HOLE pixels in the overlap, FEATHER-blend the two
  tiles' predictions (linear ramp) to avoid any 1-channel discontinuity (research: even small seams ring
  globally). Outside overlap, single tile. We only ever write HOLE pixels (non-hole keeps true DATA), so
  seams only matter where a hole straddles the boundary — the feather handles exactly that.
- Units go 2016 → ~4032 (×2). Generalises to real data / wider bands as `ceil(n_chan/512)` tiles.

## Files that must change (scope before coding)
1. `data_preparation/simulated/extract_patches_sim.py` — add the freq-tiling loop; store per-unit
   `freq_lo` (native channel offset within the band), `native_n_chan` = tile width, and
   `freq_min_patch`/`freq_max_patch` per tile (for tile-aware positional encoding — see #6).
   The real extractor `data_preparation/real/extract_variants.py` ALREADY tiles freq and stores
   `freq_lo`/`freq_min_patch`/`freq_max_patch` — reuse its schema so sim and real finally match.
2. `inference/inpaint_write.py` — currently uses GLOBAL `hf.attrs['chan_lo']` for the channel slice.
   Use `chan_lo + per-unit freq_lo`, and `native_n_chan` = tile width. Implement overlap feather-blend
   (accumulate per baseline, or read-modify-write with a feather weight). THE TRICKIEST PART.
3. `evaluation/set_holes_flag.py` and `evaluation/set_holes_weight.py` — same per-unit `freq_lo` offset.
4. `inference/oracle_level0.py`, `repr_diag.py`, `oracle_phasefix.py` — use per-unit `freq_lo` so they
   keep working on tiled h5. Re-run `repr_diag.py` to confirm the freq error drops near zero with tiling
   BEFORE retraining (cheap, no GPU).
5. `evaluation/compare_images.py` — ADD a delay/power-spectrum metric (FFT along frequency of the
   per-baseline visibilities; inpainted vs clean vs flagged) as the HEADLINE; continuum off-source RMS +
   RMSE-vs-clean become a does-no-harm check. Also add CLASSICAL baselines (DPSS least-squares + 1D delay
   CLEAN at minimum; GPR if time) so the diffusion fill is benchmarked against them, not just flagging.
6. **Positional encoding (important):** `inference/inpaint_infer.py` builds PE once from the GLOBAL
   `band_min`/`band_max`. For tiles, PE must reflect each tile's OWN range (`freq_min_patch`/
   `freq_max_patch`), else the model can't tell where in the band a tile sits. Check how
   `train.py`/`data.py` build PE; make it per-tile-range-aware and consistent train↔inference.

## Validate cheaply BEFORE the retrain (the gate)
- Re-extract sim run1 with tiling (build on `data_preparation/simulated/jobs/reextract.sh`), re-run
  `repr_diag.py`: confirm "amp resize roundtrip" error drops from 0.115 toward ~0.
- Run a phase-fix-style oracle on the TILED h5 and image it (reuse `oracle_phasefix.py` + `image_eval.sh`
  + the new delay metric): does the perfect-fill ceiling beat flagging in DELAY SPACE (headline) and do no
  harm on continuum? Only if yes does the retrain make sense.
- Run the tiling delay-space ablation: oracle WITH vs WITHOUT tiling, compared in delay space. The tiled
  oracle's high-delay power must match the non-tiled oracle's — else feathering is injecting seam power.

## Then retrain (only after the gate passes)
- Re-extract all training runs run[1-9] with tiling via `reextract.sh` (check which `run<N>/sim_clean.ms`
  still exist — /scratch3 auto-deletes at 90 days; missing → full `simulate.sh`). Keep SEED.
- Retrain: `model/sim/train_sim.sh` (full-amp → phase1_all) and/or `train_sim_decompose.sh` (smooth
  target, recommended per research). Back up existing checkpoints first.
- Infer → write-back → image; judge on continuum RMS AND delay space vs flagging.

## Hard constraints / gotchas
- ALL execution is on the ilifu SLURM cluster; the USER runs every cluster command (agent has no working
  SSH). Write code, commit LOCALLY, give the user exact push/pull/sbatch commands.
- Push to `main` is classifier-blocked for the agent — commit locally; the USER pushes.
- Containers: torch only in `ASTRO-GPU-PyTorch-2026-01-28.sif`; casacore+skimage only in
  `ASTRO-PY3.10.sif`; wsclean in `oxkat-0.41.sif`. Singularity runs via srun/sbatch only.
- Verify cluster-specific detail against `.claude/docs/` — never guess.
- Sim run1: MS `/scratch3/users/$USER/rfi/simulated/run1/sim_clean.ms`, H5 `.../run1/dataset.h5`.
- Code style: human-researcher, no AI-sounding comments/docstrings, imports at top, minimal comments.
  Test/diagnostic scripts log device + milestones + per-iter progress (flushed).
- Anchors (sim run1 imaging): clean 1.262e-4, flagged 2.461e-4, current-oracle 3.107e-4,
  phase-fix-oracle 2.192e-4 (RMS) / RMSE 2.562e-4 / peak overshoot 7%.

## Deliver FIRST (before any code)
Present for sign-off: (a) exact tiling geometry (tile spans, overlap, ownership, feather window) for the
898-ch band; (b) the per-unit h5 schema you'll standardise on; (c) the write-back blending algorithm;
(d) the PE-per-tile plan; (e) the delay-space metric definition.
