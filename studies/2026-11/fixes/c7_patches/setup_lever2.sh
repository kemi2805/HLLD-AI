#!/bin/bash
# Calea HLLD worktree for lever 2 (user 2026-11-07: "Do these two levers. Be
# ambitious. But watch out that the plan does not change"): HLLD-fix13 = 51e9c67 +
# lev2_hlld.diff (fix12's late rung and rescue harvest + RMHD_MERGE_SWEEPS: one
# seven-wave pipeline call per RK stage, off by default) + lev2_files/
# (sweep_merge.py, tests).  rmhd: rmhd_final-loose11 (lever 1).  Uncommitted.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/HLLD
git worktree add --detach $B/HLLD-fix13 51e9c67
cd $B/HLLD-fix13
git apply --check $S/lev2_hlld.diff
git apply $S/lev2_hlld.diff
cp -r $S/lev2_files/src $S/lev2_files/tests .
mkdir -p data
echo "HLLD: $(git rev-parse --short HEAD) + $(git diff --stat | tail -1)"
grep -c "_rendezvous" src/physics/exact_flux.py
grep -c "_merged_sweeps" src/physics/driver2d.py
