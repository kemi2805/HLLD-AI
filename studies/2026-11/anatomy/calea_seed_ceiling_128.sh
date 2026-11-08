#!/bin/bash
#SBATCH --job-name=seedceil-r2d128
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Why the five-wave planar solver answers 8% of the four-quadrant 128^2 run
# (anatomy study, 2026-11-04): HLLD scripts/seed_ceiling.py on that run's
# harvest -- the seven-wave Newton, one attempt, production's budget, started
# from the run's networks (ft3, s42, s47), from the known answer moved by
# steps of 0 .. 0.1, and from ft3's seed with its error shrunk; plus the
# seven-wave residual AT the known answer (constructible? a root?).  Groups:
# first (ft3 converged), retry (s42/s47 converged), planar (the planar solver).
# The 2026-10-03 rotor measurement predates the closed-form Alfven rotation.
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
HD=$BASE/HLLD-harvest
PY=$HOME/venv/rmhd/bin/python
OUT=$BASE/HLLD/results/r2d_retrain/anatomy128
mkdir -p "$OUT"
cd "$HD" || exit 1
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') || exit 2
. "$RMHD/scripts/kernels.env"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=32 TORCH_THREADS=8
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
"$PY" -u scripts/seed_ceiling.py $BASE/HLLD/results/r2d_128_exact_ml_r2d_it3/harvest \
    --n 1500 --radius 0 0.001 0.003 0.01 0.03 0.1 --draws 2 --shrink 0.5 0.25 0.1 \
    --ckpts $BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt,data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt \
    --out "$OUT/seed_ceiling.npz" || exit 3
echo "== $(date) done"
