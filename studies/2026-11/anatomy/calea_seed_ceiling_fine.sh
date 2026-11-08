#!/bin/bash
#SBATCH --job-name=seedceil-fine
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The small start radii of the seed ceiling on the four-quadrant 128^2 run
# (78408 printed radii with %.2f, so 0, 1e-3 and 3e-3 overwrote each other):
# the same draw (seed 20261003, 1500 per group), radii 0, 3e-4, 1e-3.
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
HD=$BASE/HLLD-harvest
S=/mnt/rafast/miler/scratch/r2d_retrain/anatomy
PY=$HOME/venv/rmhd/bin/python
OUT=$BASE/HLLD/results/r2d_retrain/anatomy128
cd "$HD" || exit 1
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') || exit 2
. "$RMHD/scripts/kernels.env"
export HLLD_DIR=$HD PYTHONPATH=$HD
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=32 TORCH_THREADS=8
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
"$PY" -u "$S/seed_ceiling_fine.py" $BASE/HLLD/results/r2d_128_exact_ml_r2d_it3/harvest \
    --n 1500 --radius 0 0.0003 0.001 --draws 2 --shrink 0.5 \
    --ckpts $BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt \
    --out "$OUT/seed_ceiling_fine.npz" || exit 3
echo "== $(date) done"
