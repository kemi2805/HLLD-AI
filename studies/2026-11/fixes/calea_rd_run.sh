#!/bin/bash
#SBATCH --job-name=rd-run
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Production's answers with a slow jump across the Alfven speed (2.0% of the
# 4Q 128^2 run sample, verify/run) solved again as regular seven-wave answers
# (rotation in front of the slow wave, fixes/rd_solve_run.py); user
# 2026-11-05: "Are all faces now solved with the seven wave family?"
set -u
S=/mnt/rafast/miler/scratch/fixes
BASE=/mnt/rafast/miler/codes/rotor2d
PY=$HOME/venv/rmhd/bin/python
export RMHD_DIR=$BASE/rmhd_final-loose5 PYTHONPATH=$BASE/rmhd_final-loose5:$S
export SP_POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz FIX_DIR=$S
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
. "$BASE/rmhd_final-loose5/scripts/kernels.env"
V=$BASE/HLLD/results/fixes/verify/run
OUT=$BASE/HLLD/results/fixes/rd_run
mkdir -p "$OUT"
cd "$S" || exit 1
echo "== $(date) $(hostname)"
for f in "$V"/chunk_???.npz; do
    b=$(basename "$f" .npz)
    "$PY" -u rd_solve_run.py "$f" "$OUT/$b.npy" > "$OUT/$b.log" 2>&1 &
done
wait
cat "$OUT"/chunk_*.log | grep -v -i warn > "$OUT/all.log"
echo "== $(date) done"
cat "$OUT/all.log" | tail -n 40
