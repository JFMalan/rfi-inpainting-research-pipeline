import sys
import numpy as np

ms_path = sys.argv[-1]

# Radiometer equation: sigma = SEFD / sqrt(2 * delta_nu * delta_t)
# MeerKAT L-band SEFD ~430 Jy, channel width 208984 Hz, dump time 8s
sefd = 430.0
delta_nu = 208984.0
delta_t = 8.0
sigma = sefd / np.sqrt(2.0 * delta_nu * delta_t)  # ~0.105 Jy per visibility

sm.openfromms(ms_path)
sm.setnoise(mode='simplenoise', simplenoise=f'{sigma}Jy')
sm.corrupt()
sm.close()

print(f"added thermal noise sigma={sigma:.4f} Jy to {ms_path}")
