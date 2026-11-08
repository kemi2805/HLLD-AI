#!/bin/bash
# Calea rmhd worktree for lever 1 (user 2026-11-07: "Do these two levers. Be
# ambitious. But watch out that the plan does not change"): rmhd_final-loose11 =
# 9ca5a4b + lev_rmhd.diff (the late rung, the one-call option, and the compiled
# twins of the planar root-finders behind RMHD_PLANAR_NJIT, off by default) +
# lev_files/ (coplanar7_b.py, planar_njit.py, tests).  HLLD stays HLLD-fix12.
# Uncommitted.  Nothing running uses this worktree.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/rmhd_final
git worktree add --detach $B/rmhd_final-loose11 9ca5a4b
cd $B/rmhd_final-loose11
git apply --check $S/lev_rmhd.diff
git apply $S/lev_rmhd.diff
cp -r $S/lev_files/rmhd $S/lev_files/tests .
mkdir -p data
for f in $B/rmhd_final-loose10/data/*; do cp -P "$f" data/; done
echo "rmhd: $(git rev-parse --short HEAD) + $(git diff --stat | tail -1)"
grep -c "RMHD_PLANAR_NJIT" rmhd/batched/planar_njit.py
grep -c "_PNJ" rmhd/batched/planar4_b.py rmhd/batched/shock_b.py rmhd/batched/slow_shock_b.py
