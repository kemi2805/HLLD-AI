#!/bin/bash
#SBATCH --job-name=admfix-validate
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The loose-fallback fix of the slow-shock admissibility filter (rmhd_final-
# admiss, uncommitted, 2026-11-04): (1) every leaked slow shock of the three
# filter runs (flagged_adm64/*_admiss.npz) re-solved, numpy and njit -- each
# must now FAIL or find a root, never REPRO (the stored grid cell) unless the
# unfiltered solver also has no root; (2) the 7,525-face replay with the filter
# + fix, numpy and njit, which must agree BITWISE, and against the pre-fix
# replays (C_filter, C_filter_njit); (3) the whole rmhd_final suite (want:
# 205 + 6 new filter tests passed, 3 skipped, calea's 3 known failures).
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
A=$B/HLLD/results/r2d_retrain/planar_audit
O=$A/replay
RF=$B/rmhd_final-admiss
PY=$HOME/venv/rmhd/bin/python
FL=$A/admiss_flagged_1103.npz
AL=$B/HLLD/results/r2d_retrain/it1/all.npz
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ADMISSIBLE RMHD_SLOWSHOCK_ADMISSIBLE RMHD_SLOWSHOCK RMHD_SHOCK RMHD_FAN RMHD_ALFVEN RMHD_ML_CKPT RMHD_ML_CKPTS
export PYTHONPATH=$RF HLLD_DIR=$B/HLLD-admiss
echo "== $(date) $(hostname) rmhd $(git -C $RF rev-parse --short HEAD) (+patch: $(grep -c discarded $RF/rmhd/batched/slow_shock_b.py) 'discarded' lines)"
"$PY" -c "import rmhd; print('== rmhd from', rmhd.__file__)"

echo "== (1) leaked slow shocks re-solved"
for m in numpy njit; do
    echo "## RMHD_SLOWSHOCK=$m"
    RMHD_SLOWSHOCK=$m OMP_NUM_THREADS=1 "$PY" -u "$S/fixcheck_loose.py" "$A/flagged_adm64" 1000000000 2>&1
done

echo "== (2) 7,525-face replay, filter + fix"
. "$RF/scripts/kernels.env"
export RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt
export RMHD_POOL=63 RMHD_POOL_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
cd "$HLLD_DIR" || exit 1
RMHD_SLOWSHOCK=numpy RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=report "$PY" -u "$S/r2d_admiss_replay.py" "$FL" "$AL" \
    "$O/C_fix_numpy" 2>&1 | grep -E "^==|^   (flagged|control)"
RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=report "$PY" -u "$S/r2d_admiss_replay.py" "$FL" "$AL" \
    "$O/C_fix_njit" 2>&1 | grep -E "^==|^   (flagged|control)"
"$PY" - "$O" <<'PY'
import sys
import numpy as np
O = sys.argv[1]
z = {k: np.load("%s/%s/faces.npz" % (O, k)) for k in ("C_fix_numpy", "C_fix_njit", "C_filter", "C_filter_njit", "Anp_report")}
def cmp(a, b):
    A, Bz = z[a], z[b]
    same = (A["F"] == Bz["F"]).all(axis=1)
    return "flux bitwise %d / %d; exact %d -> %d (equal %s); rung equal %s" % (
        same.sum(), same.size, A["exact"].sum(), Bz["exact"].sum(),
        bool((A["exact"] == Bz["exact"]).all()), bool((A["rung"] == Bz["rung"]).all()))
print("== fix: numpy vs njit:", cmp("C_fix_numpy", "C_fix_njit"))
print("== pre-fix filter (numpy) vs fix (numpy):", cmp("C_filter", "C_fix_numpy"))
print("== pre-fix filter (njit) vs fix (njit):", cmp("C_filter_njit", "C_fix_njit"))
print("== production (numpy) vs fix (numpy):", cmp("Anp_report", "C_fix_numpy"))
g = z["C_fix_numpy"]["group"]
for k in ("Anp_report", "C_filter", "C_fix_numpy"):
    e = z[k]["exact"]
    print("   %-12s exact: flagged %d / %d, control %d / %d" % (
        k, e[g == 0].sum(), (g == 0).sum(), e[g == 1].sum(), (g == 1).sum()))
print("== wall: fix numpy %.0f s, fix njit %.0f s (pre-fix njit %.0f s)" % (
    float(z["C_fix_numpy"]["wall"]), float(z["C_fix_njit"]["wall"]), float(z["C_filter_njit"]["wall"])))
PY

echo "== (3) rmhd_final suite (clean env, 64 workers)"
cd "$RF" || exit 1
LOG=$B/HLLD/logs/admfix_pytest_${SLURM_JOB_ID:-0}.log
env -u RMHD_FAN -u RMHD_ALFVEN -u RMHD_SLOWSHOCK -u RMHD_SHOCK -u RMHD_ML_CKPT -u RMHD_ML_CKPTS \
    -u RMHD_POOL -u RMHD_POOL_THREADS -u NUMBA_NUM_THREADS -u HLLD_DIR OMP_NUM_THREADS=1 \
    "$PY" -m pytest -q -n 64 -rfEX -p no:cacheprovider > "$LOG" 2>&1
st=$?
tail -12 "$LOG"
echo "== pytest exit $st; failures (calea's known three: test_batching_pays_off_at_production_batch_size, test_no_warm_start_is_needed, test_alfven_recovers_the_constructed_state[LA]):"
awk '/^=+ short test summary info =+$/ { s = 1; next } s && /^(FAILED|ERROR|XPASS) / { print "   " $2 }' "$LOG"
echo "== $(date) done; whole pytest output $LOG"
