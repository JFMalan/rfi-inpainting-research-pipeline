# Tricolour RFI Flagging — Handoff for Independent Review

## Context

This is an Honours research project building a conditional DDPM (Palette architecture) for inpainting
RFI-corrupted MeerKAT radio telescope spectrograms. The pipeline extracts 256×256 time-frequency
patches from real MeerKAT L-band data (900–1650 MHz) to use as training data.

RFI flagging is done with **tricolour** on the raw Measurement Set before patch extraction. The flags
are the ground-truth mask used by the inpainting model — so accuracy matters a lot. Under-flagging
leaves RFI in training patches that the model treats as clean signal. Over-flagging destroys usable
data and reduces dataset size.

The dataset is observation `1525469431_sdp_l0.ms` on the ilifu cluster. One scan (~300 time bins,
~3900 frequency channels, ~120 cross-correlation baselines) is used for testing.

---

## Current State of the Pipeline

### Files

- `data_preparation/real/tricolour-flagging.yaml` — tricolour strategy config (reproduced below)
- `data_preparation/real/extract_ms.py` — reads MS, applies divisive normalisation per baseline, extracts 256×256 patches to HDF5
- `data_preparation/real/visualisation/visualise_real.py` — generates diagnostic plots from MS + HDF5
- `data_preparation/real/jobs/flag_real_test.sh` — SLURM test job (Devel partition, scan 1 only)

### Current tricolour-flagging.yaml

```yaml
strategies:
    -
        name: flag_nans_zeros
        task: flag_nans_zeros
    -
        name: background_static_mask
        task: apply_static_mask
        kwargs:
            accumulation_mode: "or"
            uvrange: ""
    -
        name: background_flags
        task: sum_threshold
        kwargs:
            outlier_nsigma: 8
            windows_time: [1, 2, 4, 8]
            windows_freq: [1, 2, 4, 8]
            background_reject: 2.0
            background_iterations: 5
            spike_width_time: 12.5
            spike_width_freq: 10.0
            time_extend: 0
            freq_extend: 0
            freq_chunks: 10
            average_freq: 1
            flag_all_time_frac: 1.0
            flag_all_freq_frac: 1.0
            rho: 1.3
            num_major_iterations: 5
    -
        name: residual_flag
        task: uvcontsub_flagger
        kwargs:
            major_cycles: 7
            or_original_from_cycle: 0
            taylor_degrees: 10
            sigma: 9.0
    -
        name: final_st_very_broad
        task: sum_threshold
        kwargs:
            outlier_nsigma: 8
            windows_time: [1, 2, 4, 8]
            windows_freq: [32, 48, 64, 128]
            background_reject: 2.0
            background_iterations: 5
            spike_width_time: 6.5
            spike_width_freq: 64.0
            time_extend: 0
            freq_extend: 0
            freq_chunks: 10
            average_freq: 1
            flag_all_time_frac: 1.0
            flag_all_freq_frac: 1.0
            rho: 1.3
            num_major_iterations: 1
    -
        name: final_st_broad
        task: sum_threshold
        kwargs:
            outlier_nsigma: 8
            windows_time: [1, 2, 4, 8]
            windows_freq: [1, 2, 4, 8]
            background_reject: 2.0
            background_iterations: 5
            spike_width_time: 6.5
            spike_width_freq: 10.0
            time_extend: 0
            freq_extend: 0
            freq_chunks: 10
            average_freq: 1
            flag_all_time_frac: 1.0
            flag_all_freq_frac: 1.0
            rho: 1.3
            num_major_iterations: 1
    -
        name: final_st_narrow
        task: sum_threshold
        kwargs:
            outlier_nsigma: 7
            windows_time: [1, 2, 4, 8]
            windows_freq: [1, 2, 4, 8]
            background_reject: 2.0
            background_iterations: 5
            spike_width_time: 2
            spike_width_freq: 2.0
            time_extend: 0
            freq_extend: 0
            freq_chunks: 10
            average_freq: 1
            flag_all_time_frac: 1.0
            flag_all_freq_frac: 1.0
            rho: 1.3
            num_major_iterations: 1
    -
        name: flag_autos
        task: flag_autos
    -
        name: combine_with_input_flags
        task: combine_with_input_flags
```

---

## Observed Problems (from diagnostic plots)

All plots are in `vis-real/` in the project root. Read them before making recommendations.

### 1. Three clusters of missed narrow-band RFI (confirmed from `flag_diagnostics_zoom.png`)

These appear as clear spikes above the local continuum in the **unflagged** mean spectrum, with
**no corresponding flagged mean** — meaning tricolour is not touching them at all:

| Frequency | Description |
|-----------|-------------|
| ~1083–1095 MHz | cluster of sharp narrow spikes, ~20–40% above local continuum |
| ~1140 MHz | single sharp spike |
| ~1378–1385 MHz | broad hump, 5–10% above continuum |
| ~1515–1520 MHz | sharp spike just below the 1525 MHz static mask boundary |

### 2. Mean flag fraction ~48.6% (from `flag_per_baseline.png`, `flag_cdf.png`)

The CDF shows ~48% of channels flagged >90% of the time. This is **uniform across all 120 baselines**
(flag_per_baseline.png shows a flat bar at ~0.486 with near-zero variance). This means it is
frequency-domain flagging, not antenna-specific.

The cause is the three known persistent MeerKAT RFI bands (930–960 MHz, 1170–1300 MHz, 1525–1630 MHz)
being fully flagged by `apply_static_mask`. Those bands collectively cover ~240 MHz of the 750 MHz
range (~32%), and uvcontsub_flagger + SumThreshold contributes additional channel flags in the
960–1170 MHz transition region.

This 48.6% figure is likely **correct** — those bands are genuinely unusable. Whether this is
acceptable for training depends on the research goal.

### 3. Patch histogram shows three distinct populations (`patch_flag_hist.png`)

- 0–7% flag fraction: clean band patches (900–960 clean, 1300–1525 clean)
- 13–15%: patches overlapping a band edge
- 38–40%: patches covering 1170–1300 MHz persistent RFI band

Mean patch flag fraction = 0.116. Zero patches exceed 50% flags (the extract step caps these).

### 4. Patches look visually clean where unflagged (`patches_hdf5/page_001.png` etc.)

The flagged bands are correctly identified (solid green). The unflagged patches show realistic
thermal noise and fringe structure. The diagonal stripe pattern visible in some patches is real
interferometric fringe rotation, not artefacts.

---

## What Has Already Been Tried

| Change | Result |
|--------|--------|
| `time_extend: 1` / `freq_extend: 1` | Massive over-flagging (65% per time bin), reverted to 0 |
| `sigma: 7.0` in uvcontsub_flagger | Individual pixel false positives, diffuse flagging histogram, reverted |
| `sigma: 11.0` | Best result so far, but still misses the clusters above |
| `sigma: 9.0` | Current value (user-set), between the two — not yet fully evaluated |
| `taylor_degrees: 20` | Too high — absorbs broad RFI into bandpass model, misses it entirely |
| `taylor_degrees: 6` | Too low — can't fit the 750 MHz MeerKAT bandpass, blanket over-flags |
| `taylor_degrees: 10` | Current value — reasonable tradeoff |
| `flag_all_time_frac: 0.6` (not 1.0) | Caused horizontal band over-flagging; reverted to 1.0 (disabled) |
| `spike_width_freq: 2.0` in final_st_narrow | Partially helped narrow spikes; some still missed |

---

## Key tricolour Concepts

**sum_threshold:** Estimates a 2D background by iterative sigma-clipping (controlled by
`background_reject`, `background_iterations`), then flags residuals above `outlier_nsigma` × local
RMS. `spike_width_time`/`spike_width_freq` sets the expected RFI morphology — a spike much narrower
than `spike_width_freq` is harder to distinguish from noise.

**uvcontsub_flagger:** Fits a Chebyshev polynomial of degree `taylor_degrees` to each baseline's
spectrum per time slot, subtracts it, then clips residuals at `sigma` × MAD. If `taylor_degrees` is
too high, broad RFI structures get absorbed into the polynomial fit and disappear before thresholding.
If `sigma` is too low and the number of time bins is small (~300), the MAD estimate is noisy and
thermal noise peaks exceed threshold by chance.

**apply_static_mask:** Uses tricolour's built-in `4k_lband_meerkat.staticmask` — a binary mask for
known persistent MeerKAT L-band RFI bands. Applied before SumThreshold so the statistics are not
contaminated by the persistent emitters.

**flag_all_time_frac / flag_all_freq_frac:** If set below 1.0, tricolour will blanket-flag an entire
row or column when the flagged fraction exceeds the threshold. Setting to 1.0 disables this completely
(only individual detections survive).

**time_extend / freq_extend:** Morphological dilation applied after detection. Setting to 0 gives
pixel-specific flags (desired for this project). Setting to 1 dilates by 1 pixel in each direction,
which rapidly over-flags at the 300-time-bin scale used here.

---

## Research Goal for Flagging

The flags are used as the **inpainting mask** — the model is trained to reconstruct the RFI-corrupted
pixels from the clean surroundings. The ideal flag mask:

1. Flags every pixel that contains RFI (no missed spikes in training patches)
2. Does not flag clean thermal noise pixels (no false positives that remove good signal)
3. Is pixel-specific — not entire rows or columns unless the whole row/column is genuinely RFI

The downstream effect of under-flagging is worse than over-flagging for this task, because
under-flagged RFI pixels get treated as clean signal during training.

---

## Question for the Reviewer

Given everything above, what is the best approach to flag the four missed spike clusters
(1083–1095, 1140, 1378–1385, 1515–1520 MHz) without introducing false positives in the clean bands?

Options to consider:
1. **Custom static mask** for the known narrow persistent emitters — deterministic, no threshold risk
2. **Lower `outlier_nsigma` in `final_st_narrow`** from 7 to 5 or 6 — risks noise false positives
3. **Add a dedicated narrow-spike SumThreshold pass** with `spike_width_freq: 1`, `windows_freq: [1, 2]`, higher `background_iterations`
4. **Lower `sigma` in uvcontsub_flagger** below 9.0 — risks individual pixel false positives at 300 time bins
5. **Something else entirely**

Provide a concrete YAML change recommendation with reasoning, not just a direction.
