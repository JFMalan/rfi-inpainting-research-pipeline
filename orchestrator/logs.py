import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configlib import load_experiment, REPO
from submit import job_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment')
    ap.add_argument('stages', nargs='*', help='show these stages regardless of state')
    ap.add_argument('--tail', type=int, default=30, help='lines per log (0 = whole file)')
    ap.add_argument('--all', action='store_true', help='every stage with a log, not just RUNNING')
    ap.add_argument('--err', action='store_true', help='stderr logs instead of stdout')
    args = ap.parse_args()

    exp, _ = load_experiment(args.experiment)
    state_path = REPO / 'logs' / f"pipeline_{exp['name']}.json"
    if not state_path.exists():
        print(f'no state file at {state_path} — nothing submitted yet')
        return
    state = json.loads(state_path.read_text())

    stream = 'stderr' if args.err else 'stdout'
    bar = '=' * 100
    shown = 0
    for name, rec in state.items():
        jobid = rec.get('jobid', '?')
        s, elapsed = job_info(jobid)
        if args.stages:
            if name not in args.stages:
                continue
        elif not args.all and s != 'RUNNING':
            continue
        logs = glob.glob(str(REPO / 'logs' / f'*-{jobid}-{stream}.log'))
        if not logs:
            continue
        log = Path(logs[0])
        print(f'\n{bar}\n  {name}  job {jobid}  {s}  {elapsed}\n  {log}\n{bar}')
        lines = log.read_text(errors='replace').splitlines()
        if args.tail and len(lines) > args.tail:
            print(f'  ... ({len(lines) - args.tail} earlier lines)')
            lines = lines[-args.tail:]
        print('\n'.join(lines))
        shown += 1

    if not shown:
        print('no matching logs (nothing RUNNING?) — try --all or name a stage')


if __name__ == '__main__':
    main()
