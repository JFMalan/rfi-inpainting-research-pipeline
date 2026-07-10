# Figures pipeline

One generator per report figure, regenerated from checkpoints/metrics on disk. Every script
prints flushed progress (paths found, row/tile counts) so a SLURM log shows it's alive.

| Report figure (final.md spec) | Generator | Required inputs | Pipeline stage that produces the inputs |
|---|---|---|---|
| Dataset examples per run (input / RFI mask / clean target), simulated | `figures/visualise_simulate.py --input --output` | `dataset.h5` for one sim run | `data_preparation/simulated/extract_patches_sim.py` (via `inject_rfi.py` + `add_noise.py`) |
| Dataset examples, real MeerKAT | `figures/visualise_real.py --ms --output` | raw/flagged MS | tricolour flagging stage on the target MS (`data_preparation/real/tricolour-flagging.yaml`) |
| Training curves (loss + val metrics vs epoch, mean-fill line) | `figures/train_curves.py --run-dir --out` (**new**) | `<run_dir>/log.json` | `model/train.py` (phase 1) or `model/train_real.py` (phase 2) — see log format below |
| Noise-threshold / recoverability curve | `figures/plot_noise_threshold.py --runs-root --scales --out` | one `phase1_thr_n*/log.json` per noise scale | repeated `model/train.py` runs from the noise-threshold sweep (`inference/jobs` / lecturer noise-gen sweep) |
| Fill-check panels (observed / target / smooth fill / matched-grain fill / delay spectra) | `model/diagnostics/noise_free_fill_check.py --h5 --output` (**lives outside figures/ — needs a GPU + checkpoint, not moved in the reorg**) | extracted `dataset.h5` + trained checkpoint | `data_preparation/simulated` extraction + `model/train.py` checkpoint |
| Continuum image comparisons with residuals | `evaluation/compare_images.py --clean --flagged --meanfill --classical --gpr --inpainted --out --metrics-out` (**lives outside figures/**) | per-variant FITS images | `evaluation/image_eval.sh` (wsclean imaging of each MS variant after write-back) |
| Delay-spectrum comparisons | `figures/plot_delay_npz.py --npz --out` | `fakehole_delay_eval` output npz (`truth`/`flagged`/`dpss`/`gpr`/`model_nf*` 1-D arrays) | `evaluation/fakehole_delay_eval.py` |
| Massoud ladder chart | `figures/massoud_ladder.py --results r0.json r1.json ... --labels R0 R1 ... --out` (**new**) | one `metrics.json` per rung from `evaluation/evaluate.py`, all scored on the same held-out test run | `model/train.py` (rung training, `configs/experiment/massoud_r{0..3}.yaml`) then `evaluation/evaluate.py` |
| Flag-fraction / RFI-width crossover plot | `evaluation/plot_width_sweep.py --root --widths --out` (**lives outside figures/**) | `metrics.json` per width from `evaluation/compare_images.py` | `evaluation/rfi_width_sweep` machinery + `compare_images.py` |
| Selective-inpaint panels (fill / kept-flagged overlays) | `model/diagnostics/compare_models_real.py --keep-persist` (**lives outside figures/ — needs a GPU + checkpoint**) | real extraction variant h5 + checkpoint(s) | `data_preparation/real/extract_variants.py` + `model/train_real.py` checkpoint |

No figure in the spec list is missing a generator. Three of the nine already lived outside
`figures/` before this pass (`evaluation/compare_images.py`, `evaluation/plot_width_sweep.py`,
`model/diagnostics/{noise_free_fill_check,compare_models_real}.py`) because they need a GPU/MS/
checkpoint to run, unlike the plot-from-disk scripts under `figures/`. They are left in place —
the table above is the map, not a relocation.

## Sample/prediction panel generators (feed `visualise_samples.py`/`visualise_real_inpaint.py`)

- `figures/inpaint_viz.py` — runs a checkpoint over held-out sim or real tiles, saves an npz.
- `figures/visualise_samples.py` — renders that npz (or a `train.py` `sample_e*.npz`) into panels.
- `figures/visualise_real_inpaint.py` — same, for real-data (`obs`/`real_flags`/`fake_mask`/`pred`) npz.
- `figures/plot_noise_data.py`, `figures/plot_noise_samples.py` — panels across the noise-scale sweep.
- `figures/viz_writeback.py` — MS write-back sanity check (src vs `INPAINTED_DATA` column, per baseline).

## Training log format (`model/train.py` and `model/train_real.py`)

Both write `<out_dir>/log.json` — a JSON list, one dict appended per epoch, rewritten in full
each epoch (`json.dumps(log, indent=2)`). Every entry has `epoch`, `loss`, `sec`, `lr`. On
evaluated epochs (`(epoch+1) % sample_every == 0`, or the last epoch) it also gets:

- **phase 1** (`train.py`): `complex_mae`, `complex_mf`, `amp_mae`, `amp_mf`, `psnr`,
  `phase_err`, `nfr`, `beats_mf`. `*_mf` are the per-epoch mean-fill baseline (per-patch mean of
  the known pixels, same channels/mask).
- **phase 2** (`train_real.py`): `complex_mae`, `tre`, `fake_mae`, `mf_fake_mae`, `nfr`,
  `beats_mf`. `mf_fake_mae` is the mean-fill baseline scored on the fake (self-supervised) holes.

`figures/train_curves.py` tells the two apart by the presence of `tre` in an evaluated entry
and plots whichever metric set is present, with the `*_mf`/`mf_*` values drawn as a horizontal
reference line (mean over evaluated epochs).

## Massoud ladder input format

`evaluation/evaluate.py` already writes `<out>/metrics.json` per run (no `--json-out` flag was
needed — it wasn't stdout-only). Relevant keys: `psnr_mean` (their metric) and
`complex_mae_mean` (ours), both scored on the same `--split test` held-out run. Feed one such
`metrics.json` per rung into `figures/massoud_ladder.py --results r0.json r1.json r2.json r3.json
--labels R0 R1 R2 R3`.

Caveat: rungs R0/R1 train `--amp-only` (1-channel output, no phase), so `metrics.complex_mae`
short-circuits to `0.0` for them (`pred.shape[1] < 3`) — it is not a real zero-error score.
`massoud_ladder.py` detects this (`complex_mae_mean == 0 and phase_err_mean == 0`) and draws
those bars as `N/A (amp-only)` rather than plotting a misleading zero.
