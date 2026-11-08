#!/bin/bash
#SBATCH --job-name=adm64-aud-r2d
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The 4Q treatment harvest only, as soon as its run ended (the post job redoes all three).
# Admissibility audit (admiss_harvest.py); the control is r2d_64_exact_ssretry in admiss_harvests_1103.json.
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
PY=$HOME/venv/rmhd/bin/python
cd "$B/rmhd_final" || exit 1
. scripts/kernels.env
export OMP_NUM_THREADS=1 NPROC=60
echo "== $(date) $(hostname)"
"$PY" -u /mnt/rafast/miler/scratch/r2d_retrain/planar_audit/admiss_harvest.py \
    "$R/r2d_retrain/planar_audit/admiss_harvests_adm64_r2d.json" \
    "$R/r2d_64_exact_admiss"
echo "== $(date) done"
