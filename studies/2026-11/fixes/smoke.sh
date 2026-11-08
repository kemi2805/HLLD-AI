#!/bin/bash
# Local smoke test of replay_pop.py on 10 lanes: fix off vs on.
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=/Users/miler/Codes/HLLD PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2 OMP_NUM_THREADS=1 RMHD_POOL=0
PY=/Users/miler/Codes/py-work/bin/python
RMHD_SLOWSHOCK_LOOSE_RETRY=0 $PY -u $F/replay_pop.py $F/smoke_pop.npz $F/smoke_off.npz --chunk 16
RMHD_SLOWSHOCK_LOOSE_RETRY=1 $PY -u $F/replay_pop.py $F/smoke_pop.npz $F/smoke_on.npz --chunk 16
