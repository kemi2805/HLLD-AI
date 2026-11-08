#!/bin/bash
#SBATCH --job-name=adm64-xflag
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# extract_flagged.py over the six paired 64^2 harvests (controls + filter runs):
# the flagged rows with per-wave diagnostics, and the RH certificate of every
# slow shock (2026-11-04: are the filter runs' remaining flags non-roots?).
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
PY=$HOME/venv/rmhd/bin/python
cd "$B/rmhd_final" || exit 1
. scripts/kernels.env
export OMP_NUM_THREADS=1 NPROC=60
echo "== $(date) $(hostname)"
"$PY" -u /mnt/rafast/miler/scratch/r2d_retrain/planar_audit/extract_flagged.py \
    "$R/r2d_retrain/planar_audit/flagged_adm64" \
    "$R/r2d_64_exact_ctl1102" "$R/r2d_64_exact_admiss" \
    "$R/ot_64_exact_ssretry" "$R/ot_64_exact_admiss" \
    "$R/rotor_64_exact_ssretry" "$R/rotor_64_exact_admiss"
echo "== $(date) done"
