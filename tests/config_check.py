import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'orchestrator'))
import configlib

# Gate: the login-node fallback parser must agree with PyYAML on every config file.
# Run inside ASTRO-PY3.10.sif (has PyYAML); fails loudly if the parsers diverge or
# any config no longer loads/validates.

def main():
    if configlib._pyyaml is None:
        print('PyYAML not available here - run inside ASTRO-PY3.10.sif', flush=True)
        sys.exit(2)

    cfg_dir = configlib.REPO / 'configs'
    files = sorted(cfg_dir.rglob('*.yaml'))
    bad = 0
    for f in files:
        text = f.read_text()
        ref = configlib.parse_yaml(text)
        configlib._pyyaml, saved = None, configlib._pyyaml
        try:
            fb = configlib.parse_yaml(text)
        finally:
            configlib._pyyaml = saved
        status = 'ok' if fb == ref else 'MISMATCH'
        bad += status != 'ok'
        print(f'{status:8s} {f.relative_to(configlib.REPO)}', flush=True)

    for exp in sorted((cfg_dir / 'experiment').glob('*.yaml')):
        configlib.load_experiment(exp)
        print(f'loads    {exp.relative_to(configlib.REPO)} (validation passed)', flush=True)

    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
