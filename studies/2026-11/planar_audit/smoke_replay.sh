#!/bin/bash
# local smoke: 40 faces through the patched HLLD path, report then refuse
set -u
cd /Users/miler/Codes/claude_scratch/r2d_retrain/planar_audit
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export PYTHONPATH=/Users/miler/Codes/rmhd_final/.claude/worktrees/stoic-swanson-701e4a
export HLLD_DIR=/Users/miler/Codes/HLLD/.claude/worktrees/admiss
export RMHD_ML_CKPT=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export OMP_NUM_THREADS=4
PY=/Users/miler/Codes/py-work/bin/python
mkdir -p smoke
# the 40 faces: no all.npz locally, so build a tiny stand-in from rows_1103 (controls = all0/all1 rows)
$PY - <<'PY'
import numpy as np
z = np.load("rows_1103.npz", allow_pickle=True)
UL = np.concatenate([z["all0_U_L"][:200], z["all1_U_L"][:200]]); UR = np.concatenate([z["all0_U_R"][:200], z["all1_U_R"][:200]])
np.savez("smoke/all_standin.npz", U_L=UL, U_R=UR, gamma=5.0/3.0)
PY
for mode in report 1; do
  RMHD_ADMISSIBLE=$mode $PY -u replay_driver_smoke.py admiss_flagged_1103.npz smoke/all_standin.npz smoke/out_$mode --n-control 10 --max 40 2>&1 | grep -v Warning | grep -v "^  W\|^  return\|^  b2\|^  f =" | tail -8
done
