#!/bin/bash
# Local smoke test of r2d_resolve.py (2026-11-02): solved / lost / perturb on a
# few faces of one shard of the 4Q production harvest, then harvest_to_dataset.
set -eu
cd /Users/miler/Codes/claude_scratch/r2d_retrain/smoke
PY=/Users/miler/Codes/py-work/bin/python
RF=/Users/miler/Codes/rmhd_final
export HLLD_DIR=/Users/miler/Codes/HLLD
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_POOL
export RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt
. "$RF/scripts/kernels.env"
rm -rf out_solved out_lost out_perturb
$PY -u ../r2d_resolve.py solved harvest out_solved --max 1500 2>&1 | tail -n 3
$PY -u ../r2d_resolve.py lost harvest out_lost --max 300 2>&1 | tail -n 3
(cd "$RF" && $PY scripts/harvest_to_dataset.py /Users/miler/Codes/claude_scratch/r2d_retrain/smoke/out_solved/harvest --out /Users/miler/Codes/claude_scratch/r2d_retrain/smoke/solved.npz 2>&1 | tail -n 2)
$PY -u ../r2d_resolve.py perturb solved.npz out_perturb --n 1100 2>&1 | tail -n 3
ls out_solved/harvest out_perturb/harvest | head -20
