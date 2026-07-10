import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t0 = time.time()

EPILOG = """
input format:
  each --results file is the metrics.json written by evaluation/evaluate.py
  (python evaluate.py --data <test_run_h5> --ckpt <rung>/best.pt --out <rung_eval_dir>),
  run once per Massoud-ladder rung on the SAME held-out test run. relevant keys:
    psnr_mean        - their metric (PSNR, dB, higher is better)
    complex_mae_mean - ours (complex-visibility MAE, lower is better)
  amp-only rungs (R0/R1, trained with --amp-only) have no phase channels, so
  evaluate.py's complex_mae comes back 0.0 (metrics.complex_mae short-circuits when
  pred has <3 channels) — this script detects that and plots those bars as N/A
  rather than a misleading zero.
"""


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


def main(args):
    labels = args.labels if args.labels else [Path(p).stem for p in args.results]
    if len(labels) != len(args.results):
        raise SystemExit("--labels must match --results in count")

    psnr, cplx, cplx_na = [], [], []
    for p, lab in zip(args.results, labels):
        d = json.loads(Path(p).read_text())
        log(f"{lab}: {p}  psnr_mean={d.get('psnr_mean', float('nan')):.2f}  "
            f"complex_mae_mean={d.get('complex_mae_mean', float('nan')):.4f}  "
            f"phase_err_mean={d.get('phase_err_mean', float('nan')):.4f}")
        psnr.append(d.get('psnr_mean', np.nan))
        amp_only = d.get('complex_mae_mean', 0.0) == 0.0 and d.get('phase_err_mean', 0.0) == 0.0
        cplx.append(np.nan if amp_only else d.get('complex_mae_mean', np.nan))
        cplx_na.append(amp_only)

    x = np.arange(len(labels))
    w = 0.36
    fig, ax1 = plt.subplots(figsize=(1.6 * len(labels) + 3, 5.5))
    ax1.bar(x - w / 2, psnr, w, color='tab:blue', label='PSNR (their metric)')
    ax1.set_ylabel('PSNR (dB, higher = better)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel('Massoud component-ladder rung')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w / 2, np.nan_to_num(cplx), w, color='tab:red', label='complex MAE (ours)')
    ax2.set_ylabel('complex MAE (lower = better)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    for xi, na, b in zip(x, cplx_na, bars2):
        if na:
            b.set_height(0)
            ax2.text(xi + w / 2, ax2.get_ylim()[1] * 0.03, 'N/A\n(amp-only)',
                     ha='center', va='bottom', fontsize=7, color='tab:red')

    for xi, v in zip(x - w / 2, psnr):
        if np.isfinite(v):
            ax1.text(xi, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8, color='tab:blue')
    for xi, v, na in zip(x + w / 2, cplx, cplx_na):
        if not na and np.isfinite(v):
            ax2.text(xi, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color='tab:red')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    ax1.set_title(args.title or 'Massoud component ladder: their metric (PSNR) vs ours (complex MAE)')
    fig.tight_layout()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches='tight')
    log(f"saved -> {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', nargs='+', required=True, help='r0.json r1.json r2.json ...')
    ap.add_argument('--labels', nargs='+', default=None, help='R0 R1 R2 ... (default: filename stems)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default=None)
    main(ap.parse_args())
