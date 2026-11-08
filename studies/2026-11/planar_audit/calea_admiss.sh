#!/bin/bash
#SBATCH --job-name=r2d-admiss
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Admissibility (entropy, Lax, slow-vs-fast order) of every exact answer the
# 4Q 64^2 run harvested (planar acceptance study, 2026-11-03).  Read-only on
# results/r2d_retrain except results/r2d_retrain/planar_audit/.
set -u
B=/mnt/rafast/miler/codes/rotor2d
PY=$HOME/venv/rmhd/bin/python
cd "$B/rmhd_final" || exit 1
. scripts/kernels.env
export OMP_NUM_THREADS=16
echo "== $(date) $(hostname) rmhd_final $(git rev-parse --short HEAD)"
O=$B/HLLD/results/r2d_retrain/planar_audit
"$PY" -u /mnt/rafast/miler/scratch/r2d_retrain/planar_audit/admiss_all.py "$B/HLLD/results/r2d_retrain" \
    "$O/admiss_flagged_1103.npz" "$O/admiss_1103.json"
echo "== $(date) exit $?"
