#!/bin/bash
# local: a short 2D run with the sweeps merged (RMHD_MERGE_SWEEPS=1) against the same run unmerged:
# restart state, diag.csv and harvest files must agree bit for bit.  run_merge_test.sh PROBLEM N STEPS [extra env...]
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
PROB=$1; N=$2; STEPS=$3; shift 3
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPT=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=2 TORCH_THREADS=2 RMHD_POOL=${RMHD_POOL:-0}
for kv in "$@"; do export "$kv"; done
cd $F/hlld_fix
T=$F/levers/merge_test/${PROB}_${N}_$(echo "$@" | tr ' =' '_-')
rm -rf $T; mkdir -p $T
for m in 0 1; do
    t0=$(date +%s)
    RMHD_MERGE_SWEEPS=$m /Users/miler/Codes/py-work/bin/python -u scripts/run_2d.py --problem $PROB --n $N --solver exact \
        --tend 0.4 --nsnap 2 --tau-weak 1e-2 --tau-bt 1e-9 --exact-retries 2 --exact-max-iter 40 \
        --out $T/m$m --harvest $T/m$m/harvest --harvest-all --restart-every $STEPS --log-every 1 \
        --max-steps $STEPS > $T/m$m.log 2>&1
    echo "merge=$m: exit $?, $(( $(date +%s) - t0 )) s; $(grep -E '^\s+step' $T/m$m.log | tail -1)"
done
/Users/miler/Codes/py-work/bin/python - $T <<'PY'
import glob, os, sys
import numpy as np
T = sys.argv[1]
def same_npz(a, b):
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    if sorted(za.files) != sorted(zb.files):
        return False, "keys differ"
    for k in za.files:
        x, y = za[k], zb[k]
        if x.dtype.kind == "f":
            if not (x.shape == y.shape and np.array_equal(x, y, equal_nan=True)):
                return False, k
        elif not np.array_equal(x, y):
            return False, k
    return True, ""
ok, why = same_npz(T + "/m0/restart.npz", T + "/m1/restart.npz")
print("restart state bitwise:", ok, why)
da = [l.split(",") for l in open(T + "/m0/diag.csv")]
db = [l.split(",") for l in open(T + "/m1/diag.csv")]
print("diag.csv rows identical:", da == db, "(%d rows)" % len(da))
fa = sorted(os.path.basename(f) for f in glob.glob(T + "/m0/harvest/*.npz"))
fb = sorted(os.path.basename(f) for f in glob.glob(T + "/m1/harvest/*.npz"))
print("harvest files", len(fa), "same names:", fa == fb)
bad = [f for f in fa if not same_npz(T + "/m0/harvest/" + f, T + "/m1/harvest/" + f)[0]]
print("harvest files differing:", bad[:5], len(bad))
PY
