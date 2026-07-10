import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configlib import load_experiment


def _fmt(v):
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, float):
        return f'{v:g}'
    return str(v)


def _draw(exp, run, knob, lo, hi):
    # deterministic per (experiment seed, run, knob); the test run pins noise at 1.0 elsewhere
    rng = random.Random(f"{exp['seed']}:{run}:{knob}")
    return round(rng.uniform(lo, hi), 4)


def _sefd_nodes(tel):
    pairs = zip(tel['sefd']['nodes_mhz'], tel['sefd']['nodes_jy'])
    return ' '.join(f'{m}:{j}' for m, j in pairs)


def stage_env(exp, tel, stage, run=None, mode=None, arena=None, variant=None):
    sim, band, sky = exp['sim'], tel['band'], tel['sky']
    runs_root = exp['paths']['runs']
    p1_out = f"{runs_root}/{exp['name']}_phase1"

    if stage == 'simulate':
        if run is None:
            raise SystemExit('simulate needs --run <N|test>')
        test = run == 'test'
        seed = exp['sim']['seed_base'] + (0 if test else int(run))
        rfi = sim['rfi']
        env = {
            'RUN_ID': run,
            'SEED': seed,
            'GEN_RANDOM_SKY': sim['random_sky'],
            'SYNTHESIS': sim['synthesis_h'],
            'NCHAN': sim['nchan'],
            'IMG_SIZE': sim['img_size'],
            'NOISE_SCALE': 1.0 if test else _draw(exp, run, 'noise', *sim['noise_scale']),
            'TARGET_FRAC': _draw(exp, run, 'flag_frac', *rfi['target_flag_frac']),
            'PERSIST_FRAC': _draw(exp, run, 'persist', *rfi['persist_frac']),
            'RFI_SCALE_MIN': rfi['scale'][0],
            'RFI_SCALE_MAX': rfi['scale'][1],
            'FLUX_MIN': sky['flux_min_jy'],
            'FLUX_MAX': sky['flux_max_jy'],
            'DIR': sky['pointing_str'],
            'TEL_MODEL': tel['simms_model'],
            'POLS': tel['polarizations'],
            'DUMP_T': tel['dump_time_s'],
            'F0_MHZ': band['sim_f0_mhz'],
            'BW_MHZ': band['sim_bandwidth_mhz'],
            'FREQ_MIN': band['extract_min_mhz'],
            'FREQ_MAX': band['extract_max_mhz'],
            'SEFD_NODES': _sefd_nodes(tel),
            'SMOOTH_BINS': exp['extract']['smooth_bins'],
            'MAX_BL_FLAG': exp['extract']['sim_max_bl_flag_frac'],
            'CLEAN_TARGET': exp['extract']['clean_target'],
        }
        return env

    if stage in ('flag_real', 'extract_real'):
        real = exp['real']
        return {
            'MS': real['ms'],
            'FIELD': real['field'],
            'COLUMN': real['column'],
            'MAX_BL_FLAG': real['max_bl_flag_frac'],
            'FREQ_MIN': band['extract_min_mhz'],
            'FREQ_MAX': band['extract_max_mhz'],
            'SMOOTH_BINS': exp['extract']['smooth_bins'],
            'IMG_SIZE': sim['img_size'],
        }

    if stage == 'train_phase1':
        t = exp['train']['phase1']
        env = {
            'RUN_ID': 'all',
            'PHASE': 1,
            'TAG': exp['name'],
            'EPOCHS': t['epochs'],
            'BATCH': t['batch'],
            'LR': t['lr'],
            'SEED': exp['seed'],
            'VAL_EVAL_STEPS': t['val_eval_steps'],
            'VAL_EVAL_PATCHES': t['val_eval_patches'],
            'CLEAN_TARGET': t.get('clean_target', exp['extract']['clean_target']),
        }
        # Massoud-ladder rung knobs (absent on the production experiment)
        for key, var in (('amp_only', 'AMP_ONLY'), ('raw_amp', 'RAW_AMP'),
                         ('rand_mask', 'RAND_MASK'), ('loss', 'LOSS')):
            if key in t:
                env[var] = t[key]
        return env

    if stage == 'train_phase2':
        if mode not in ('finetune', 'scratch'):
            raise SystemExit('train_phase2 needs --mode finetune|scratch')
        t = exp['train']['phase2']
        env = {
            'MODE': mode,
            'EPOCHS': t['epochs'],
            'BATCH': t['batch'],
            'LR': t['lr'],
            'MAX_PATCHES': exp['real']['max_samples'],
            'MAX_FLAG_FRAC': exp['real']['max_sample_flag_frac'],
            'SEED': exp['seed'],
        }
        if mode == 'finetune':
            env['INIT_FROM'] = f'{p1_out}/best.pt'
        if t['ema_decay'] != 'auto':
            env['EMA_DECAY'] = t['ema_decay']
        return env

    if stage == 'infer':
        if arena not in ('continuum', 'delay'):
            raise SystemExit('infer needs --arena continuum|delay')
        inf = exp['inference']
        return {
            'STEPS': inf['steps'],
            'BATCH': inf['batch'],
            'NOISE_FLOOR': inf['noise_floor'][arena],
            'TAG': exp['name'],
        }

    if stage == 'writeback':
        wb = exp['writeback']
        return {
            'OUTCOL': wb['out_col'],
            'RESET_COL': wb['reset_col'],
            'NO_FEATHER': not wb['feather'],
            'KEEP_PERSIST': 1 if variant == 'inpaint_selective' else 0,
        }

    if stage == 'image_eval':
        im = exp['eval']['imaging']
        return {
            'IMSIZE': im['imsize'],
            'CELL': im['cell'],
            'NITER': im['niter'],
            'WEIGHT': im['weight'],
            'AUTO_MASK': im['auto_mask'],
            'AUTO_THRESHOLD': im['auto_threshold'],
            'MGAIN': im['mgain'],
            'DPSS_HW': exp['eval']['delay']['dpss_hw'],
            'DPSS_LAM': exp['eval']['delay']['dpss_lam'],
            'KEEP_PERSIST': 1 if variant == 'inpaint_selective' else 0,
        }

    if stage == 'delay_eval':
        d = exp['eval']['delay']
        return {
            'DPSS_HW': d['dpss_hw'],
            'DPSS_LAM': d['dpss_lam'],
            'GPR_ELL': d['gpr_ell'],
            'GPR_NOISE': d['gpr_noise'],
            'FG_BINS': d['fg_bins'],
            'BOOTSTRAP': d['bootstrap'],
            'MAX_FLAG_FRAC': exp['real']['max_sample_flag_frac'],
            'NOISE_FLOOR': exp['inference']['noise_floor']['delay'],
        }

    raise SystemExit(f'unknown stage: {stage}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment')
    ap.add_argument('--stage', required=True)
    ap.add_argument('--run')
    ap.add_argument('--mode')
    ap.add_argument('--arena')
    ap.add_argument('--variant')
    args = ap.parse_args()

    exp, tel = load_experiment(args.experiment)
    env = stage_env(exp, tel, args.stage, run=args.run, mode=args.mode,
                    arena=args.arena, variant=args.variant)
    for k, v in env.items():
        print(f'{k}="{_fmt(v)}"')


if __name__ == '__main__':
    main()
