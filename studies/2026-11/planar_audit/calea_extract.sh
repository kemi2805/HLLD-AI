#!/bin/bash
#SBATCH --job-name=r2d-extract
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Extract the audit's flagged rows + control samples (planar acceptance study,
# 2026-11-03).  Read-only on results/r2d_retrain except the one output file.
set -u
B=/mnt/rafast/miler/codes/rotor2d
PY=$HOME/venv/rmhd/bin/python
echo "== $(date) $(hostname)"
"$PY" -u /mnt/rafast/miler/scratch/r2d_retrain/planar_audit/extract_rows.py \
    "$B/HLLD/results/r2d_retrain" "$B/HLLD/results/r2d_retrain/planar_audit/rows_1103.npz"
echo "== $(date) exit $?"
