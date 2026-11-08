#!/bin/bash
#SBATCH --job-name=r2d-audit
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Audit of the four-quadrant retrain data (r2d_audit.py; user 2026-11-03:
# "Are we only using proper training and validation data for the Q4 ml?  Did
# we check the eos and all the states?  How many datapoints do we have").
# Read-only on results/r2d_retrain; writes audit_1103.json there.
set -u
B=/mnt/rafast/miler/codes/rotor2d
PY=$HOME/venv/rmhd/bin/python
cd "$B/rmhd_final" || exit 1
. scripts/kernels.env
export OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32
echo "== $(date) $(hostname) rmhd_final $(git rev-parse --short HEAD)"
"$PY" -u /mnt/rafast/miler/scratch/r2d_retrain/r2d_audit.py "$B/HLLD/results/r2d_retrain" \
    --json "$B/HLLD/results/r2d_retrain/audit_1103.json"
echo "== $(date) exit $?"
