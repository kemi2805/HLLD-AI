#!/bin/bash
# Re-apply the rmhd patch in rotor2d/rmhd_final-admiss from scratch (git only;
# login node): the 2026-11-04 version -- the slow-shock admissibility filter
# (slow_shock_b + slow_shock_njit) WITH the loose-fallback fix (no grid cell
# handed back where the filter discarded a root) and its tests.
set -eu
W=/mnt/rafast/miler/codes/rotor2d/rmhd_final-admiss
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
cd "$W"
git checkout -- rmhd/batched/slow_shock_b.py rmhd/batched/slow_shock_njit.py
rm -f tests/batched/test_slow_shock_admissible.py
git apply "$S/slow_shock_admissible_1104.diff"
git apply "$S/test_slow_shock_admissible_1104.diff"
git status --short
git diff --stat
grep -c "discarded" rmhd/batched/slow_shock_b.py rmhd/batched/slow_shock_njit.py
