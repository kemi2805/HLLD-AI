#!/bin/bash
#SBATCH --job-name=r2d-admhv
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Admissibility of every harvested exact answer of the three 64^2 problems,
# just before (alfx1, 10-26/27) and after (ssretry, 10-29/30) the slow-shock
# retry became the default (planar acceptance study, 2026-11-03).  Read-only.
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
PY=$HOME/venv/rmhd/bin/python
cd "$B/rmhd_final" || exit 1
. scripts/kernels.env
export OMP_NUM_THREADS=1 NPROC=60
echo "== $(date) $(hostname) rmhd_final $(git rev-parse --short HEAD)"
"$PY" -u /mnt/rafast/miler/scratch/r2d_retrain/planar_audit/admiss_harvest.py \
    "$R/r2d_retrain/planar_audit/admiss_harvests_1103.json" \
    "$R/r2d_64_exact_alfx1" "$R/r2d_64_exact_ssretry" "$R/ot_64_exact_alfx1" "$R/ot_64_exact_ssretry" \
    "$R/rotor_64_exact_alfx1" "$R/rotor_64_exact_ssretry"
echo "== $(date) exit $?"
