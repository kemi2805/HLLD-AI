#!/bin/bash
# local: RMHD_HARVEST_RESCUES on the 24-face ledger smoke (late rung, it3 network, as run_late.sh)
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RETRIES=2 OMP_NUM_THREADS=1 RMHD_POOL=0
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1 RMHD_PLANAR4_ROT=1
export RMHD_COPLANAR7=late RMHD_PLANAR4_STAR_BATCH=1 RMHD_HARVEST_RESCUES=1 RMHD_COPLANAR7_CALLS=${CALLS:-1}
rm -rf $F/planar_ml/smoke_harvest
/Users/miler/Codes/py-work/bin/python -u $F/planar_ml/harvest_smoke.py $F/smoke_c7_pop.npz $F/planar_ml/smoke_harvest 2>&1 | grep -v -i warn
