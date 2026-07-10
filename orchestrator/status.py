import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configlib import load_experiment, REPO
from submit import job_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment')
    args = ap.parse_args()

    exp, _ = load_experiment(args.experiment)
    state_path = REPO / 'logs' / f"pipeline_{exp['name']}.json"
    if not state_path.exists():
        print(f'no state file at {state_path} — nothing submitted yet')
        return
    state = json.loads(state_path.read_text())

    print(f"{'stage':<28} {'jobid':<10} {'state':<12} log")
    for name, rec in state.items():
        jobid = rec.get('jobid', '?')
        s = job_state(jobid)
        logs = glob.glob(str(REPO / 'logs' / f'*-{jobid}-stdout.log'))
        log = logs[0] if logs else '-'
        print(f'{name:<28} {jobid:<10} {s:<12} {log}')


if __name__ == '__main__':
    main()
