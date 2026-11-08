#!/bin/bash
#SBATCH --job-name=adm-validate
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The njit port of the slow-shock admissibility filter (rmhd_final-admiss,
# uncommitted): (1) the slow-shock tests, new and old, in a clean environment;
# (2) the 7,525-face replay with the COMPILED filter, which must reproduce the
# numpy-filter replay (C_filter, job 77251) bit for bit.
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
O=$B/HLLD/results/r2d_retrain/planar_audit/replay
RF=$B/rmhd_final-admiss
PY=$HOME/venv/rmhd/bin/python
FL=$B/HLLD/results/r2d_retrain/planar_audit/admiss_flagged_1103.npz
AL=$B/HLLD/results/r2d_retrain/it1/all.npz
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ADMISSIBLE RMHD_SLOWSHOCK_ADMISSIBLE RMHD_SLOWSHOCK RMHD_SHOCK RMHD_FAN RMHD_ALFVEN
export PYTHONPATH=$RF HLLD_DIR=$B/HLLD-admiss
echo "== $(date) $(hostname) rmhd $(git -C $RF rev-parse --short HEAD) (+patch)"
cd "$RF" || exit 1
"$PY" -m pytest -q -p no:cacheprovider tests/batched/test_slow_shock_admissible.py \
    tests/batched/test_slow_shock_njit.py tests/batched/test_slow_shock_retry.py \
    tests/batched/test_shock_njit.py 2>&1 | tail -5
. "$RF/scripts/kernels.env"
export RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt
export RMHD_POOL=63 RMHD_POOL_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$HLLD_DIR" || exit 1
RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=report "$PY" -u "$S/r2d_admiss_replay.py" "$FL" "$AL" \
    "$O/C_filter_njit" 2>&1 | grep -E "^==|^   (flagged|control)"
"$PY" - "$O" <<'PY'
import sys
import numpy as np
O = sys.argv[1]
a = np.load(O + "/C_filter/faces.npz")
b = np.load(O + "/C_filter_njit/faces.npz")
same = (a["F"] == b["F"]).all(axis=1)
print("== njit filter vs numpy filter: flux bitwise on %d / %d faces; exact equal %s; rung equal %s"
      % (same.sum(), same.size, bool((a["exact"] == b["exact"]).all()), bool((a["rung"] == b["rung"]).all())))
print("== wall: numpy %.0f s, njit %.0f s" % (float(a["wall"]), float(b["wall"])))
PY
echo "== $(date) done"
