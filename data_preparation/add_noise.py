import sys
import numpy as np

# CASA passes args as: casa -c script.py <ms_path>
# sys.argv[0] is the script name; the MS path is the next positional arg
ms_path = None
for arg in sys.argv[1:]:
    if not arg.startswith('-'):
        ms_path = arg
        break
if ms_path is None:
    raise RuntimeError("MS path not found in argv")

tb.open(ms_path + '/SPECTRAL_WINDOW')
delta_nu = float(tb.getcol('CHAN_WIDTH')[0, 0])
tb.close()

sefd = 430.0
delta_t = 8.0
sigma = sefd / np.sqrt(2.0 * delta_nu * delta_t)

sm.openfromms(ms_path)
sm.setnoise(mode='simplenoise', simplenoise=f'{sigma}Jy')
sm.corrupt()
sm.close()

print(f"delta_nu={delta_nu:.1f} Hz  sigma={sigma:.4f} Jy  ms={ms_path}")
