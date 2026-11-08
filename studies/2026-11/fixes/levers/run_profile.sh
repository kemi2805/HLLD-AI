#!/bin/bash
# local: profile the late chain on hard faces (production networks, the late runs' switches, in process)
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export OMP_NUM_THREADS=1 RMHD_POOL=0 NUMBA_NUM_THREADS=2 TORCH_THREADS=2
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1 RMHD_PLANAR4_ROT=1
export RMHD_COPLANAR7=late RMHD_PLANAR4_STAR_BATCH=1
/Users/miler/Codes/py-work/bin/python -u $F/levers/profile_rung.py $F/levers/profile_faces.npz "$@" 2>&1 | grep -v -i "warn\|W = xp.sqrt"
