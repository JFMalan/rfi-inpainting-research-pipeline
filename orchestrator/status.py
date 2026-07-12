import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configlib import load_experiment, REPO
from submit import job_info


def job_dependency(jobid):
    try:
        out = subprocess.run(['scontrol', 'show', 'job', str(jobid)],
                             capture_output=True, text=True, timeout=30)
        m = re.search(r'Dependency=(\S+)', out.stdout)
        dep = m.group(1) if m else ''
        return '' if dep in ('', '(null)') else dep
    except Exception:
        return ''


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
    stage_of = {str(rec.get('jobid')): name for name, rec in state.items()}

    print(f"{'stage':<28} {'jobid':<10} {'state':<24} {'elapsed':<12} log")
    for name, rec in state.items():
        jobid = rec.get('jobid', '?')
        s, elapsed = job_info(jobid)
        if s == 'PENDING':
            elapsed = '-'
            dep = job_dependency(jobid)
            waits = [stage_of.get(j, j) for j in re.findall(r'\d+', dep)]
            s = f"(P) <- {','.join(waits)}" if waits else '(P)'
        logs = glob.glob(str(REPO / 'logs' / f'*-{jobid}-stdout.log'))
        log = logs[0] if logs else '-'
        print(f'{name:<28} {jobid:<10} {s:<24} {elapsed:<12} {log}')


if __name__ == '__main__':
    main()
