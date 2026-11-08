#!/bin/bash
#SBATCH --job-name=anatomy-table
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The anatomy of the four-quadrant 128^2 exact run with ft3 (user 2026-11-04:
# "an analysis on what faces succeed and fail, and why they fail.  And what
# faces use the 5 wave solver and why"): every attempted face with its
# position, time, outcome, states and answer (face_table.py), then a
# stratified sample for the laptop (sample_table.py).
set -u
S=/mnt/rafast/miler/scratch/r2d_retrain/anatomy
RUN=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_128_exact_ml_r2d_it3
OUT=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/anatomy128
PY=$HOME/venv/rmhd/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "$OUT/faces"
echo "== $(date) $(hostname)"
"$PY" -u "$S/face_table.py" "$RUN" "$OUT/faces" --procs 56 || exit 3
echo "== $(date) table done"
"$PY" -u "$S/sample_table.py" "$OUT/faces" "$OUT/sample.npz" || exit 4
echo "== $(date) done"
