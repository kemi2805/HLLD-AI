#!/bin/bash
#SBATCH --job-name=r2d-admrep-numpy
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Planar acceptance study (2026-11-03): replay the 4Q 64^2 run's inadmissible
# faces (2,525 problems, admiss_all.py) + 5,000 unflagged controls through the
# production path, in uncommitted worktrees (HLLD-admiss, rmhd_final-admiss).
#   njit : A = RMHD_ADMISSIBLE=report (production decisions, counts only)
#          B = RMHD_ADMISSIBLE=1     (refuse inadmissible answers at acceptance)
#   numpy: A_np = report with the numpy slow shock (C's baseline)
#          C    = RMHD_SLOWSHOCK_ADMISSIBLE=1 (admissible slow-shock roots only), report
#          CB   = C + RMHD_ADMISSIBLE=1
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
O=$B/HLLD/results/r2d_retrain/planar_audit/replay
RF=$B/rmhd_final-admiss
PY=$HOME/venv/rmhd/bin/python
FL=$B/HLLD/results/r2d_retrain/planar_audit/admiss_flagged_1103.npz
AL=$B/HLLD/results/r2d_retrain/it1/all.npz
export PYTHONPATH=$RF HLLD_DIR=$B/HLLD-admiss
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ADMISSIBLE RMHD_SLOWSHOCK_ADMISSIBLE
export RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt
export RMHD_POOL=63 RMHD_POOL_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
. "$RF/scripts/kernels.env"
mkdir -p "$O"
cd "$HLLD_DIR" || exit 1
echo "== $(date) $(hostname) HLLD $(git rev-parse --short HEAD) (+patch) rmhd $(git -C $RF rev-parse --short HEAD) (+patch)"
export RMHD_SLOWSHOCK=numpy
RMHD_ADMISSIBLE=report "$PY" -u "$S/r2d_admiss_replay.py" "$FL" "$AL" "$O/Anp_report" || exit 2
RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=report "$PY" -u "$S/r2d_admiss_replay.py" "$FL" "$AL" "$O/C_filter" || exit 3
RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=1 "$PY" -u "$S/r2d_admiss_replay.py" "$FL" "$AL" "$O/CB_filter_refuse" || exit 4
echo "== $(date) done"
