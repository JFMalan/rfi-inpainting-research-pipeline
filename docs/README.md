# Working docs index

Research notes and handovers, kept for provenance. Status is relative to the final
pipeline (noise-free clean target, delay + continuum arenas; smooth/decompose target
and the TRE metric are dead ends documented in these files).

| Doc | Status | What it holds |
|-----|--------|---------------|
| [methodology-audit.md](methodology-audit.md) | current | 2026-06-25 adversarial audit; per-question verdicts (delay-space arena, classical baselines); primary citation source |
| [methodology-audit-actions.md](methodology-audit-actions.md) | current | the audit's action list; direct predecessor of final.md |
| [phase1-science-log.md](phase1-science-log.md) | current | phase-1 training history: loss-leak bug + fix (Palette contract), DDIM eta=0 decision, bias diagnostics |
| [sampling-investigation.md](sampling-investigation.md) | current | why deterministic DDIM + post-hoc noise_floor; RePaint re-noising failure documented as a dead end |
| [tiling-design-brief.md](tiling-design-brief.md) | partial | native-512 tiling geometry (2 tiles, overlap 126ch, ownership split ch 449) — geometry current, methodology part superseded by the audit |
| [ms-writeback-plan.md](ms-writeback-plan.md) | partial | write-back design: inverse transforms, safety assertions, write-only-holes decision — built as designed; column names evolved |
| [imaging-investigation-handover.md](imaging-investigation-handover.md) | partial | the oracle-vs-flagging puzzle + suspects; resolved via level-0 oracle (kept in tests/); checkpoint paths still useful |
| [research-noise-dominated-inpainting.md](research-noise-dominated-inpainting.md) | partial | information-theoretic argument amplitude noise is unrecoverable (Blau & Michaeli); superseded where it recommends decompose-then-inpaint |
| [inpainting-investigation-brief.md](inpainting-investigation-brief.md) | superseded | decompose/TRE-era investigation brief; real-data stats tables still referenced |
| [phase2-handover.md](phase2-handover.md) | superseded | decompose-era phase-2 verdict; the EMA-0.9999-froze-eval bug writeup lives here |
| [refactor-plan.md](refactor-plan.md) | superseded | 2026-06-23 metric-stack refactor, executed (noise_floor_ratio, sampling defaults); remaining items absorbed by final.md |
| [restructure-inventory.md](restructure-inventory.md) | current | the 2026-07-10 keep-vs-archive map for the final restructure |
| [restructure-layout-proposal.md](restructure-layout-proposal.md) | current | the approved target layout + config schemas |
