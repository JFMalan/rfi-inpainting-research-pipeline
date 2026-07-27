import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


PHASE1_METRICS = [('complex_mae', 'complex_mf', 'complex MAE'),
                  ('amp_mae', 'amp_mf', 'amplitude MAE'),
                  ('psnr', None, 'PSNR (dB)'),
                  ('phase_err', None, 'phase error (rad)')]
PHASE2_METRICS = [('complex_mae', None, 'complex MAE'),
                  ('fake_mae', 'mf_fake_mae', 'fake-hole MAE'),
                  ('tre', None, 'TRE'),
                  ('nfr', None, 'noise floor ratio')]


def main(args):
    run_dir = Path(args.run_dir)
    log_path = run_dir / 'log.json'
    if not log_path.exists():
        raise SystemExit(f"no log.json in {run_dir}")
    entries = json.loads(log_path.read_text())
    log(f"found {log_path}  {len(entries)} epoch records")

    ev = [e for e in entries if 'complex_mae' in e]
    is_phase2 = any('tre' in e for e in ev)
    metrics = PHASE2_METRICS if is_phase2 else PHASE1_METRICS
    log(f"format={'phase2' if is_phase2 else 'phase1'}  eval points={len(ev)}")
    if not ev:
        raise SystemExit("log.json has no evaluated epochs (no complex_mae entries)")

    epochs_all = [e['epoch'] for e in entries]
    loss_all = [e['loss'] for e in entries]
    ev_epochs = [e['epoch'] for e in ev]

    n_panels = 1 + len(metrics)
    ncols = 3
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].plot(epochs_all, loss_all, color='C0')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('train loss'); axes[0].set_title('training loss')
    axes[0].grid(alpha=0.3)

    for i, (key, mf_key, label) in enumerate(metrics, start=1):
        ax = axes[i]
        vals = [e.get(key, np.nan) for e in ev]
        ax.plot(ev_epochs, vals, 'o-', color='C1', label=label)
        if mf_key:
            mf_vals = [e[mf_key] for e in ev if mf_key in e]
            if mf_vals:
                mf_mean = float(np.mean(mf_vals))
                ax.axhline(mf_mean, color='0.4', ls='--', lw=1.3,
                           label=f'mean-fill ({mf_mean:.4f})')
        ax.set_xlabel('epoch'); ax.set_ylabel(label); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{run_dir.name} training curves", fontsize=11)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    log(f"saved -> {out}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, dest='run_dir',
                    help='training out_dir containing log.json (from train.py / train_real.py)')
    ap.add_argument('--out', required=True)
    main(ap.parse_args())
