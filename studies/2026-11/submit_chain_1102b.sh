#!/bin/bash
# 2026-11-02, after the harvester fix (one shard per call in r2d_resolve.py):
# iteration 2 again, iteration 3 after it, the 64^2 run with the chosen
# network after that.  The control (75443) is already running.
set -eu
cd /mnt/rafast/miler/scratch/r2d_retrain
J2=$(sbatch --parsable --export=ALL,NET=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/it1/ml_guess_r2d_ft1.pt calea_r2d_it2.sh)
J3=$(sbatch --parsable --dependency=afterok:$J2 calea_r2d_it3.sh)
J4=$(sbatch --parsable --dependency=afterok:$J3 calea_r2d_ml64.sh)
echo "it2 $J2  it3 $J3  ml64 $J4"
sleep 5
squeue -u miler -h -o "%i %j %N %T %r" | grep -E "r2d-"
