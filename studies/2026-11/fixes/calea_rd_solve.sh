#!/bin/bash
#SBATCH --job-name=rd-solve
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The regular seven-wave re-solve of the faces whose answer the admissibility
# gate refuses (user 2026-11-05: "take a look on the entropy fall, we may get
# this also up to 99.99%"): on each side the coplanar rotation in closed form
# (either lab sign) in front of the signed-field slow wave, Newton from the
# refused answer, every rotation combination, judged by the independent
# verifier (fixes/rd_solve.py).  One process per two lanes.
set -u
S=/mnt/rafast/miler/scratch/fixes
BASE=/mnt/rafast/miler/codes/rotor2d
PY=$HOME/venv/rmhd/bin/python
export RMHD_DIR=$BASE/rmhd_final-loose5 PYTHONPATH=$BASE/rmhd_final-loose5:$S
export SP_POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz FIX_DIR=$S
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
. "$BASE/rmhd_final-loose5/scripts/kernels.env"
OUT=$BASE/HLLD/results/fixes/rd_solve
mkdir -p "$OUT"
echo "== $(date) $(hostname)  lanes: $(wc -w < $S/rd_lanes.txt)"
cd "$S" || exit 1
set -- $(cat "$S/rd_lanes.txt")
k=0
while [ $# -gt 0 ]; do
    a=$1; shift; b=${1:-}; [ $# -gt 0 ] && shift
    "$PY" -u rd_solve.py "$OUT/rd_$k.npy" $a $b > "$OUT/rd_$k.log" 2>&1 &
    k=$((k + 1))
done
wait
cat "$OUT"/rd_*.log | grep -v -i warn > "$OUT/all.log"
echo "== $(date) done: $(grep -c '^== lane' $OUT/all.log) lanes"
