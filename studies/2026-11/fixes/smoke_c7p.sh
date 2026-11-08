#!/bin/bash
# local: 8 refused + 3 still-lost + 13 control ledger faces, final + gate + rotation, without / with the planar seven-wave stage
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2 OMP_NUM_THREADS=1 RMHD_POOL=0
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1 RMHD_PLANAR4_ROT=1
for c7 in 1; do
    RMHD_COPLANAR7=$c7 /Users/miler/Codes/py-work/bin/python -u $F/replay_pop.py $F/smoke_c7_pop.npz $F/smoke_c7_${c7}p.npz --chunk 32 2>&1 | grep -v -i warn | tail -n 2
done
