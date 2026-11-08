#!/bin/bash
# local: the late rung staged (CALLS=4, diag_4q/smoke_late.npz from run_late.sh) against one call (CALLS=1) on the 24-face ledger smoke
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2 OMP_NUM_THREADS=1 RMHD_POOL=0
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1 RMHD_PLANAR4_ROT=1
export RMHD_COPLANAR7=late
for c in 1 4; do
    t0=$(date +%s)
    RMHD_COPLANAR7_CALLS=$c /Users/miler/Codes/py-work/bin/python -u $F/replay_pop.py $F/smoke_c7_pop.npz $F/planar_ml/smoke_calls$c.npz --chunk 32 2>&1 | grep -v -i warn | tail -n 1
    echo "  calls $c: $(( $(date +%s) - t0 )) s"
done
/Users/miler/Codes/py-work/bin/python - <<'PY'
import numpy as np
F = "/Users/miler/Codes/claude_scratch/r2d_retrain/fixes"
a, b = np.load(F + "/planar_ml/smoke_calls4.npz"), np.load(F + "/planar_ml/smoke_calls1.npz")
ref = np.load(F + "/diag_4q/smoke_late.npz")
fk = [k for k in a.files if k.startswith("F_")]
for name, x in (("calls 1 vs calls 4", b), ("11-05 staged run vs calls 4", ref)):
    same = all(np.array_equal(a[k], x[k], equal_nan=True) for k in fk)
    dmax = max(float(np.nanmax(np.abs(a[k] - x[k]))) for k in fk)
    print("%-28s fluxes bitwise %s (max |dF| %.1e); src equal %s; c7 stage equal %s" % (
        name, same, dmax, np.array_equal(a["src"], x["src"]), np.array_equal(a["c7_stage"], x["c7_stage"])))
print("stages of the late answers:", np.bincount(a["c7_stage"][a["c7"]]).tolist())
PY
