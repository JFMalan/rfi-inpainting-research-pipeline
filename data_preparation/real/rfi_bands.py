import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'orchestrator'))
from configlib import parse_yaml, REPO

# Persistent-band list lives in the telescope config (oxkat/MeerKAT Cookbook
# short-baseline emitter list for the L-band instance). TEL_CONFIG selects the
# instrument; the module-level name is kept for the existing importers.

_tel = os.environ.get('TEL_CONFIG', 'meerkat_lband')
_cfg = parse_yaml((REPO / 'configs' / 'telescope' / f'{_tel}.yaml').read_text())
LBAND_PERSISTENT_MHZ = [tuple(band) for band in _cfg['persistent_rfi_mhz']]


def persist_chan_mask(freqs_mhz):
    freqs = np.asarray(freqs_mhz)
    m = np.zeros(freqs.shape, bool)
    for a, b in LBAND_PERSISTENT_MHZ:
        m |= (freqs >= a) & (freqs <= b)
    return m
