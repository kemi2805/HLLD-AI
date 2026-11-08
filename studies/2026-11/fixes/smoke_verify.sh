#!/bin/bash
# local smoke of capture_answers.py + verify_rmhd.py on the 8-lane smoke population (final2 config)
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix RMHD_DIR=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2 OMP_NUM_THREADS=1 RMHD_POOL=0
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4
PY=/Users/miler/Codes/py-work/bin/python
$PY -u $F/capture_answers.py $F/smoke_pop_b.npz $F/smoke_capture 2>&1 | grep -v -i warn | tail -n 2
$PY - <<'PYEOF'
import glob, numpy as np
F = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/smoke_capture"
parts = []
for f in sorted(glob.glob(F + "/harvest_*/solved_*.npz")) + sorted(glob.glob(F + "/planar4_*.npz")):
    z = np.load(f, allow_pickle=True)
    parts.append({k: z[k] for k in ("U_L", "U_R", "zones", "speeds")})
    print(f.split("/")[-1], len(z["U_L"]))
out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
np.savez(F + "/all.npz", gamma=5.0 / 3.0, **out)
PYEOF
$PY $F/verify_rmhd.py $F/smoke_capture/all.npz $F/smoke_capture/verify.npz 2>&1 | grep -v -i warn
