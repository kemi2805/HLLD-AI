#!/bin/bash
#SBATCH --job-name=replay-loose
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired replay of the 4Q 128^2 failure-ledger population (78407: 872 lost
# lanes + 1228 controls) through production, RMHD_SLOWSHOCK_LOOSE_RETRY off vs
# on (the loose-fallback lanes offered to the slow-shock retry; user 11-04
# "try them out").  rmhd from the worktree rmhd_final-loose (9ca5a4b + the
# patch, off = production), HLLD production (51e9c67).  The two arms run side
# by side, 32 cores each.
#   POP=... OUT=... sbatch calea_replay_loose.sh
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=${POP:-$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz}
OUT=${OUT:-$BASE/HLLD/results/fixes/replay_loose}
RMHD=$BASE/rmhd_final-loose
mkdir -p "$OUT"
export HLLD_DIR=$BASE/HLLD PYTHONPATH=$RMHD
cd "$BASE/HLLD" || exit 1
"$PY" -c 'import rmhd, sys; sys.exit(0 if "rmhd_final-loose" in rmhd.__file__ else 1)' ||
    { echo "PREFLIGHT FAILED: rmhd does not come from the worktree"; exit 2; }
grep -q LOOSE_RETRY "$RMHD/rmhd/batched/slow_shock_b.py" || { echo "PREFLIGHT FAILED: no patch"; exit 2; }
export RMHD_ML_CKPT=$BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=$BASE/rmhd_final/data/ml_guess_rotor_big_s42.pt,$BASE/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
. "$RMHD/scripts/kernels.env"
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)  rmhd $(git -C $RMHD rev-parse --short HEAD)+patch  pop $POP"
for arm in off on; do
    v=0; [ "$arm" = on ] && v=1
    RMHD_SLOWSHOCK_LOOSE_RETRY=$v RMHD_POOL=15 RMHD_POOL_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_THREADS=2 \
        "$PY" -u "$S/replay_pop.py" "$POP" "$OUT/replay_$arm.npz" --chunk 512 \
        > "$OUT/replay_$arm.log" 2>&1 &
done
wait
tail -n 2 "$OUT"/replay_*.log
echo "== $(date) done"
