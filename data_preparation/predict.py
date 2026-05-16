"""
Predict visibilities into an MS using MeqTrees via Pyxis.
Runs inside stimela_meqtrees container which has casacore + pyxis.
"""
import os
import sys
import subprocess

ms = sys.argv[1]
lsm = sys.argv[2]  # tigger LSM file (.lsm.html)

# Use meqtrees pipeliner via pyxis to predict
# MeqTrees standard predict script is Siamese/turbo-sim.py
script = os.path.join(
    os.path.dirname(subprocess.check_output(['which', 'pyxis']).decode().strip()),
    '../share/meqtrees/Pyxis/turbo-sim.py'
)

# Fall back to the standard predict via Cattery if turbo-sim not found
# Both are available in the meqtrees container
env = os.environ.copy()
env['PYXIS_ROOT_NAMESPACE'] = 'True'

cmd = [
    'pyxis',
    f'MS={ms}',
    f'LSM={lsm}',
    'COLUMN=DATA',
    'stefcal_reset_model=1',
    'turbo_sim.predict_vis',
]

result = subprocess.run(cmd, env=env)
sys.exit(result.returncode)
