#!/bin/bash
# the one failing planar4 test in three setups: today's code (numpy, njit) and the final2 code without the rotation change
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
T=tests/batched/test_planar4_start.py::test_star_start_solves_the_reversing_fans
PY=/Users/miler/Codes/py-work/bin/python
( cd $F/rmhd_fix && env -u RMHD_FAN -u RMHD_SLOWSHOCK PYTHONPATH=$F/rmhd_fix $PY -m pytest -q $T 2>&1 | tail -2 | sed 's/^/now, numpy:  /' ) &
( cd $F/rmhd_fix && . /Users/miler/Codes/rmhd_final/scripts/kernels.env && PYTHONPATH=$F/rmhd_fix $PY -m pytest -q $T 2>&1 | tail -2 | sed 's/^/now, njit:   /' ) &
( cd $F/tmp_final2 && env -u RMHD_FAN -u RMHD_SLOWSHOCK PYTHONPATH=$F/tmp_final2 $PY -m pytest -q $T 2>&1 | tail -2 | sed 's/^/final2, numpy: /' ) &
wait
