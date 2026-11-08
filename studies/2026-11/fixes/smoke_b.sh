#!/bin/bash
# Local smoke of both fixes through the production path (8 lanes): off vs on.
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2 OMP_NUM_THREADS=1 RMHD_POOL=${POOL:-0}
PY=/Users/miler/Codes/py-work/bin/python
for v in 0 1; do
    RMHD_SLOWSHOCK_LOOSE_RETRY=$v RMHD_PLANAR4_STAR=$v perl -e 'alarm 900; exec @ARGV' \
        $PY -u $F/replay_pop.py $F/smoke_pop_b.npz $F/smoke_b_$v.npz --chunk 8 2>&1 | grep -v -i warn | tail -n 1
done
