#!/bin/bash
# Re-apply the rmhd patch (slow_shock_b + slow_shock_njit) and the new test in
# rotor2d/rmhd_final-admiss from scratch (git only; login node).
set -eu
W=/mnt/rafast/miler/codes/rotor2d/rmhd_final-admiss
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
cd "$W"
git checkout -- rmhd/batched/slow_shock_b.py rmhd/batched/slow_shock_njit.py
rm -f tests/batched/test_slow_shock_admissible.py
git apply "$S/slow_shock_admissible.diff"
git apply "$S/test_slow_shock_admissible.diff"
git status --short
git diff --stat
