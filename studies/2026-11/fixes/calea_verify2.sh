#!/bin/bash
#SBATCH --job-name=verify-rmhd2
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# verify_rmhd.py again on the chunks 78642 captured / sampled (ledger answers +
# 256,000 of the 128^2 run's answers), with the corrected verifier: per-
# component physical scales, Simpson on the fan identity, TOL_RH 1e-6.
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
OUT=$BASE/HLLD/results/fixes/verify
export PYTHONPATH=$BASE/rmhd_final-loose5:$S RMHD_DIR=$BASE/rmhd_final-loose5
export SP_POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
. "$BASE/rmhd_final-loose5/scripts/kernels.env"
cd "$S" || exit 1
echo "== $(date) $(hostname): verify"
pids=()
for f in "$OUT"/ledger/chunk_???.npz "$OUT"/run/chunk_???.npz; do
    "$PY" -u verify_rmhd.py "$f" "${f%.npz}_verify.npz" > "${f%.npz}_verify.log" 2>&1 &
    pids+=($!)
done
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) done, $nf verify processes failed"
