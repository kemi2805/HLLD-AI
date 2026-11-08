#!/bin/bash
# the two failing tests, today's code vs the final2 code (no rotation / planar stage) vs the main checkout
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
T1=tests/batched/test_planar4.py::test_recovers_the_constructed_answer
T2="tests/batched/test_fullcontact.py::test_fullfuncv6_matches_scalar"
PY=/Users/miler/Codes/py-work/bin/python
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
( cd $F/rmhd_fix && PYTHONPATH=$F/rmhd_fix $PY -m pytest -q $T1 $T2 2>&1 | tail -4 | sed 's/^/now:    /' ) &
( cd $F/tmp_final2 && PYTHONPATH=$F/tmp_final2 $PY -m pytest -q $T1 $T2 2>&1 | tail -4 | sed 's/^/final2: /' ) &
( cd /Users/miler/Codes/rmhd_final && PYTHONPATH=/Users/miler/Codes/rmhd_final $PY -m pytest -q $T1 $T2 2>&1 | tail -4 | sed 's/^/main:   /' ) &
wait
