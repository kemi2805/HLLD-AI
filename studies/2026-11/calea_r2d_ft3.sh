#!/bin/bash
#SBATCH --job-name=r2d-ft3
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The four-quadrant problem at N (128, 512) with the retrained warm-start
# network ft3 (user, 2026-11-03: "do the 128² run first with ft3 and start the
# 512^2 run ... I want to be able to collect solved faces on the run from the
# 512 run").  The 64^2 run's setup (calea_r2d_ml64.sh: production's settings,
# only the primary network changed), run from the worktree HLLD-harvest =
# 51e9c67 plus patch_harvest_max.py: --harvest-max lifts the 5M-row cap so the
# harvest keeps every solved and unsolved face to the end (the solution is
# untouched).  Results and logs go to the main checkout.  Resubmitting the
# same N continues from $OUT/restart.npz (RESUME=1); the resumed process
# writes its shards as r001_*, and re-records the steps since the last
# restart file.
set -u
B=/mnt/rafast/miler/codes/rotor2d
WT=$B/HLLD-harvest
CK=$B/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt
[ -f "$CK" ] || { echo "no checkpoint $CK"; exit 2; }
: "${N:?set N}"
unset PYTHONPATH RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPTS
export PROBLEM=riemann2d N RMHD_ML_CKPT=$CK HLLD_DIR=$WT
export TAG=${TAG:-_ml_r2d_it3}
export OUT=${OUT:-$B/HLLD/results/r2d_${N}_exact$TAG}
export LOG=${LOG:-$B/HLLD/logs/r2d_${N}_exact${TAG}_$(date +%Y%m%d_%H%M%S).log}
export HARVEST_MAX=${HARVEST_MAX:-2000000000}
cd "$WT" || exit 1
exec bash scripts/calea_rotor_exact.sh
