#!/bin/bash
# 2026-11-02: the 64^2 control now; iteration 3 after iteration 2 (75279); the
# 64^2 run with the chosen network after iteration 3.
set -eu
cd /mnt/rafast/miler/codes/rotor2d/HLLD
C=$(PROBLEM=riemann2d N=64 TAG=_ctl1102 sbatch --parsable --job-name=r2d-ctl64 --time=1-00:00:00 scripts/calea_rotor_exact.sh)
cd /mnt/rafast/miler/scratch/r2d_retrain
J3=$(sbatch --parsable --dependency=afterok:75279 calea_r2d_it3.sh)
J4=$(sbatch --parsable --dependency=afterok:$J3 calea_r2d_ml64.sh)
echo "control $C  it3 $J3  ml64 $J4"
sleep 5
squeue -u miler -h -o "%i %j %N %T %r" | grep -E "r2d-(ctl64|retrain|ml64)"
