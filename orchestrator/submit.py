import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configlib import load_experiment, REPO
from resolve_config import stage_env, _fmt


def expand(p):
    return os.path.expandvars(str(p))


def build_stages(exp, tel):
    name = exp['name']
    scratch = expand(exp['paths']['scratch'])
    runs_root = expand(exp['paths']['runs'])
    p1_out = f'{runs_root}/{name}_phase1'
    p2_out = lambda m: f'{runs_root}/{name}_phase2_{m}'
    eval_out = f'{runs_root}/{name}_eval'
    sim_dir = lambda r: f'{scratch}/simulated/run{r}'
    real_h5 = f'{scratch}/real/dataset.h5'
    real_ms = expand(exp['real']['ms']) if 'real' in exp else None
    pre_flagged = exp['real'].get('pre_flagged', False)
    # write-back needs a writable MS on scratch: the tricolour path creates one while
    # flagging; the pre-flagged path copies the source (read-only /idia) verbatim
    if pre_flagged:
        writable_ms = f'{scratch}/real/{Path(real_ms).stem}.ms' if real_ms else None
    else:
        writable_ms = f'{scratch}/real/{Path(real_ms).stem}_flagged.ms' if real_ms else None
    nf = exp['inference']['noise_floor']
    d = exp['eval']['delay']
    im = exp['eval']['imaging']

    stages = []

    def add(name_, script, env, slurm, deps):
        stages.append(dict(name=name_, script=script,
                           env={k: _fmt(v) for k, v in env.items()},
                           slurm=slurm, deps=deps))

    sim_runs = [str(r) for r in range(1, exp['sim']['n_train_runs'] + 1)]
    for r in sim_runs + (['test'] if exp['sim']['test_run'] else []):
        add(f'simulate_run{r}', 'data_preparation/simulated/jobs/simulate.sh',
            stage_env(exp, tel, 'simulate', run=r), 'simulate', [])

    glob_pat = exp['train']['phase1'].get('data_glob', 'run[0-9]*')
    add('train_phase1', 'model/sim/train_sim.sh',
        {**stage_env(exp, tel, 'train_phase1'),
         'DATA': f'{scratch}/simulated/{glob_pat}/dataset.h5', 'OUT': p1_out},
        'train', [f'simulate_run{r}' for r in sim_runs])

    if pre_flagged:
        real_stage = 'extract_real'
        add('extract_real', 'data_preparation/real/jobs/extract_real.sh',
            {'MS': real_ms, 'COLUMN': exp['real']['column'],
             'MAX_BL_FLAG': exp['real']['max_bl_flag_frac'],
             'FREQ_MIN': tel['band']['extract_min_mhz'], 'FREQ_MAX': tel['band']['extract_max_mhz'],
             'SMOOTH_BINS': exp['extract']['smooth_bins'], 'IMG_SIZE': exp['sim']['img_size'],
             'OUTDIR': f'{scratch}/real', 'OUT_H5': real_h5},
            'extract', [])
        add('copy_real_ms', 'data_preparation/real/jobs/copy_ms.sh',
            {'SRC_MS': real_ms, 'DST_MS': writable_ms}, 'copy', [])
        wb_extra_deps = ['copy_real_ms']
    else:
        real_stage = 'flag_real'
        add('flag_real', 'data_preparation/real/jobs/flag_real.sh',
            {'SRC_MS': real_ms, 'FIELD_NAME': exp['real']['field'],
             'FREQ_MIN': tel['band']['extract_min_mhz'], 'FREQ_MAX': tel['band']['extract_max_mhz'],
             'MAX_BL_FLAG': exp['real']['max_bl_flag_frac'],
             'SMOOTH_BINS': exp['extract']['smooth_bins'],
             'WORKDIR': f'{scratch}/real', 'FLAGGED_MS': writable_ms,
             'PATCHES_OUT': real_h5, 'VIS_OUT': f'{scratch}/real/vis'},
            'flag', [])
        wb_extra_deps = []

    for mode in exp['train']['phase2']['modes']:
        add(f'train_phase2_{mode}', 'model/real/jobs/train_real.sh',
            {**stage_env(exp, tel, 'train_phase2', mode=mode),
             'DATA': real_h5, 'OUT': p2_out(mode),
             **({'INIT_FROM': f'{p1_out}/best.pt'} if mode == 'finetune' else {})},
            'train', [real_stage] + (['train_phase1'] if mode == 'finetune' else []))

    # post-phase-2 test-sample panels: full fill and selective (persistent bands kept flagged)
    for variant, kp in (('full', 0), ('selective', 1)):
        add(f'panels_real_{variant}', 'model/diagnostics/jobs/compare_models_real.sh',
            {'H5': real_h5, 'SIM_CKPT': f'{p1_out}/best.pt',
             'FT_CKPT': f'{p2_out("finetune")}/best.pt', 'SC_CKPT': f'{p2_out("scratch")}/best.pt',
             'KEEP_PERSIST': kp, 'NF': nf['delay'],
             'OUT': f'{eval_out}/panels_real_{variant}.png'},
            'infer', ['train_phase1', 'train_phase2_finetune', 'train_phase2_scratch'])

    for mode in exp['train']['phase2']['modes']:
        add(f'eval_delay_{mode}', 'evaluation/jobs/fakehole_delay.sh',
            {'H5': real_h5, 'CKPT': f'{p2_out(mode)}/best.pt',
             'OUT': f'{eval_out}/fakehole_delay_{mode}.npz',
             'NOISE_FLOORS': nf['delay'], 'DPSS_HW': d['dpss_hw'], 'GPR_ELL': d['gpr_ell']},
            'infer', [f'train_phase2_{mode}'])

    # PSNR/MSE/complex-MAE on the held-out test run; metrics.json feeds the ladder chart
    add('evaluate_sim', 'evaluation/jobs/eval.sh',
        {'DATA': f'{sim_dir("test")}/dataset.h5', 'CKPT': f'{p1_out}/best.pt',
         'OUT': f'{eval_out}/eval_test', 'SPLIT': 'all', 'MAX_EVAL': 512,
         'BATCH': 16, 'STEPS': 50},
        'infer', ['train_phase1', 'simulate_runtest'])

    # sim continuum arena: smooth fill (noise_floor none), test run, phase-1 model
    add('infer_sim', 'inference/jobs/inpaint_infer.sh',
        {'SIM': 1, 'H5': f'{sim_dir("test")}/dataset.h5', 'CKPT': f'{p1_out}/best.pt',
         'SMOOTH': 0, 'NOISE_FLOOR': nf['continuum'],
         'STEPS': exp['inference']['steps'], 'BATCH': exp['inference']['batch'],
         'OUTCOL': exp['writeback']['out_col'],
         'PREDS': f'{scratch}/preds_{name}_sim.npz'},
        'infer', ['train_phase1', 'simulate_runtest'])
    add('writeback_sim', 'inference/jobs/inpaint_writeback.sh',
        {'SIM': 1, 'MS': f'{sim_dir("test")}/sim_clean.ms', 'H5': f'{sim_dir("test")}/dataset.h5',
         'PREDS': f'{scratch}/preds_{name}_sim.npz', 'OUTCOL': exp['writeback']['out_col'],
         'RESET_COL': exp['writeback']['reset_col'],
         'NO_FEATHER': not exp['writeback']['feather']},
        'writeback', ['infer_sim'])
    add('image_eval_sim', 'evaluation/image_eval.sh',
        {'SIM': 1, 'MS': f'{sim_dir("test")}/sim_clean.ms', 'H5': f'{sim_dir("test")}/dataset.h5',
         'INPCOL': exp['writeback']['out_col'],
         'IMSIZE': im['imsize'], 'CELL': im['cell'], 'NITER': im['niter'],
         'MEANFILL': 1, 'DPSSFILL': 1, 'GPRFILL': 1, 'DPSS': 1, 'DELAY': 1,
         'DPSS_HW': d['dpss_hw'], 'DPSS_LAM': d['dpss_lam'],
         'GPR_ELL': d['gpr_ell'], 'GPR_NOISE': d['gpr_noise'],
         'OUT': f'{eval_out}/image_sim'},
        'image', ['writeback_sim'])

    # real arena: inpaint-everything and selective variants share one preds file
    add('infer_real', 'inference/jobs/inpaint_infer.sh',
        {'SIM': 0, 'H5': real_h5, 'CKPT': f'{p2_out("finetune")}/best.pt',
         'SMOOTH': 0, 'NOISE_FLOOR': nf['continuum'],
         'STEPS': exp['inference']['steps'], 'BATCH': exp['inference']['batch'],
         'OUTCOL': 'INPAINTED_DATA', 'PREDS': f'{scratch}/preds_{name}_real.npz'},
        'infer', ['train_phase2_finetune'])
    for variant, col, kp in (('all', 'INPAINTED_DATA', 0), ('selective', 'INPAINTED_SEL', 1)):
        add(f'writeback_real_{variant}', 'inference/jobs/inpaint_writeback.sh',
            {'SIM': 0, 'MS': writable_ms, 'H5': real_h5,
             'PREDS': f'{scratch}/preds_{name}_real.npz', 'OUTCOL': col,
             'RESET_COL': 1, 'KEEP_PERSIST': kp,
             'NO_FEATHER': not exp['writeback']['feather']},
            'writeback', ['infer_real'] + wb_extra_deps)
        add(f'image_eval_real_{variant}', 'evaluation/image_eval.sh',
            {'SIM': 0, 'MS': writable_ms, 'H5': real_h5, 'INPCOL': col,
             'IMSIZE': im['imsize'], 'CELL': im['cell'], 'NITER': im['niter'],
             'MEANFILL': 1, 'DPSSFILL': 1, 'GPRFILL': 1, 'DPSS': 1, 'DELAY': 1,
             'KEEP_PERSIST': kp, 'OUT': f'{eval_out}/image_real_{variant}'},
            'image', [f'writeback_real_{variant}'])

    wanted = exp.get('stages')
    if wanted:
        keep = set()
        for st in stages:
            if any(st['name'] == w or st['name'].startswith(w) for w in wanted):
                keep.add(st['name'])
        stages = [st for st in stages if st['name'] in keep]
        for st in stages:
            st['deps'] = [dp for dp in st['deps'] if dp in keep]
    return stages


def slurm_flags(exp, key):
    block = exp.get('slurm', {}).get(key, {})
    flags = []
    if 'partition' in block:
        flags += ['--partition', str(block['partition'])]
    if 'mem' in block:
        flags += ['--mem', str(block['mem'])]
    if 'time' in block:
        flags += ['--time', str(block['time'])]
    if 'cpus' in block:
        flags += ['--cpus-per-task', str(block['cpus'])]
    if 'gpus' in block:
        flags += [f"--gres=gpu:{block['gpus']}"]
    if 'constraint' in block:
        flags += ['--constraint', str(block['constraint'])]
    return flags


def job_state(jobid):
    try:
        out = subprocess.run(['sacct', '-j', str(jobid), '--format=State', '--noheader',
                              '--parsable2', '-X'], capture_output=True, text=True, timeout=30)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        return lines[0].split(' ')[0] if lines else 'UNKNOWN'
    except Exception:
        return 'UNKNOWN'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', help='comma-separated stage names to submit (with their deps assumed done)')
    ap.add_argument('--force', help='comma-separated stage names to resubmit even if completed')
    args = ap.parse_args()

    exp, tel = load_experiment(args.experiment)
    stages = build_stages(exp, tel)
    only = set(args.only.split(',')) if args.only else None
    force = set(args.force.split(',')) if args.force else set()

    state_path = REPO / 'logs' / f"pipeline_{exp['name']}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    jobids = {}
    resubmitted = set()
    state_cache = {}

    def cached_state(jobid):
        if jobid not in state_cache:
            state_cache[jobid] = job_state(jobid)
        return state_cache[jobid]

    for st in stages:
        name = st['name']
        if only is not None and name not in only:
            prev = state.get(name, {})
            if prev.get('jobid'):
                jobids[name] = prev['jobid']
            continue

        prev = state.get(name, {})
        dep_redone = any(dp in resubmitted for dp in st['deps'])
        if prev.get('jobid') and name not in force:
            s = job_state(prev['jobid'])
            if s == 'COMPLETED':
                print(f'skip   {name}  (job {prev["jobid"]} completed)', flush=True)
                jobids[name] = prev['jobid']
                continue
            if s == 'RUNNING' or (s == 'PENDING' and not dep_redone):
                print(f'reuse  {name}  (job {prev["jobid"]} {s.lower()})', flush=True)
                jobids[name] = prev['jobid']
                continue
            if s == 'PENDING' and dep_redone:
                # its afterok points at a dead jobid (a dep was resubmitted) — rewire
                subprocess.run(['scancel', str(prev['jobid'])], capture_output=True)
                print(f'rewire {name}  (job {prev["jobid"]} cancelled, dep resubmitted)', flush=True)
            else:
                print(f'redo   {name}  (job {prev["jobid"]} {s.lower()})', flush=True)
        elif prev.get('jobid') and name in force:
            s = job_state(prev['jobid'])
            if s in ('PENDING', 'RUNNING'):
                subprocess.run(['scancel', str(prev['jobid'])], capture_output=True)
                print(f'force  {name}  (job {prev["jobid"]} {s.lower()} cancelled)', flush=True)

        # afterok on an already-completed (possibly purged) job makes sbatch fail with
        # "Job dependency problem" — a completed dep is satisfied, so leave it out
        deps = []
        for dp in st['deps']:
            if dp not in jobids:
                continue
            jid = jobids[dp]
            if str(jid).startswith('dry_') or dp in resubmitted:
                deps.append(jid)
            elif cached_state(jid) != 'COMPLETED':
                deps.append(jid)
        missing = [dp for dp in st['deps'] if dp not in jobids]
        if missing and only is None:
            print(f'ERROR: {name} depends on unsubmitted {missing}', flush=True)
            sys.exit(1)

        cmd = ['sbatch', '--parsable'] + slurm_flags(exp, st['slurm'])
        if deps:
            cmd += [f"--dependency=afterok:{':'.join(str(j) for j in deps)}"]
        cmd += [str(REPO / st['script'])]

        env = dict(os.environ)
        env.update({k: expand(v) for k, v in st['env'].items() if v is not None})

        if args.dry_run:
            envs = ' '.join(f'{k}={v}' for k, v in sorted(st['env'].items()) if v is not None)
            print(f"DRY  {name}\n     {' '.join(cmd)}\n     env: {envs}", flush=True)
            jobids[name] = f'dry_{len(jobids)}'
            continue

        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
        if r.returncode != 0:
            print(f'ERROR submitting {name}: {r.stderr.strip()}', flush=True)
            sys.exit(1)
        jobid = r.stdout.strip().split(';')[0]
        jobids[name] = jobid
        resubmitted.add(name)
        state[name] = {'jobid': jobid, 'script': st['script']}
        state_path.parent.mkdir(exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        dep_str = f"  after {','.join(st['deps'])}" if st['deps'] else ''
        print(f'submit {name}  job {jobid}{dep_str}', flush=True)

    if not args.dry_run:
        print(f'\nstate -> {state_path}', flush=True)
        print(f"status: python3 orchestrator/status.py {args.experiment}", flush=True)


if __name__ == '__main__':
    main()
