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
