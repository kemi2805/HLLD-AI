#!/bin/bash
#SBATCH --job-name=r2d-ml64
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The four-quadrant problem at 64^2 with the retrained warm-start network
# (user, 2026-11-02: "Then please rerun 4Q with 64^2 and measure the failed
# faces and walltime").  Production's run (r2d_64_exact_ssretry, through
# scripts/calea_rotor_exact.sh: pool 32x2, retries 2, tau_weak 1e-2) with only
# the primary network changed: CKPT, or the one iteration 3 chose
# (results/r2d_retrain/it3/chosen.txt).  The retry ensemble stays rotor_big
# s42 / s47.  The control (production's network, the same code, today) is
# results/r2d_64_exact_ctl1102.
set -u
B=/mnt/rafast/miler/codes/rotor2d
CK=${CKPT:-$(cat "$B/HLLD/results/r2d_retrain/it3/chosen.txt")} || exit 2
[ -f "$CK" ] || { echo "no checkpoint $CK"; exit 2; }
unset PYTHONPATH HLLD_DIR RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPTS
export PROBLEM=riemann2d N=64 RMHD_ML_CKPT=$CK
export TAG=${TAG:-_ml_$(basename "$CK" .pt | sed 's/^ml_guess_//')}
cd "$B/HLLD" || exit 1
exec bash scripts/calea_rotor_exact.sh
