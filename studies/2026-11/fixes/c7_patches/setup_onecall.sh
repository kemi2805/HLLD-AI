#!/bin/bash
# Calea worktrees for the one-call planar seven-wave rung (RMHD_COPLANAR7_CALLS)
# and the harvest of the later rungs' answers (RMHD_HARVEST_RESCUES), both off
# by default: rmhd_final-loose10 = 9ca5a4b + c7l_rmhd.diff + coplanar7_b_onecall.py
# (+ its tests); HLLD-fix11 = 51e9c67 + hr_hlld.diff (c7l_hlld.diff + the rescue
# rows).  Uncommitted.  The running late runs keep loose9 / fix10.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/rmhd_final
git worktree add --detach $B/rmhd_final-loose10 9ca5a4b
cd $B/rmhd_final-loose10
git apply $S/c7l_rmhd.diff
cp $S/coplanar7_b_onecall.py rmhd/batched/coplanar7_b.py
cp $S/test_coplanar7_onecall.py tests/batched/test_coplanar7.py
cp $S/test_planar4_rotation.py tests/batched/test_planar4_rotation.py
mkdir -p data
for f in $B/rmhd_final-loose9/data/*; do cp -P "$f" data/; done
cd $B/HLLD
git worktree add --detach $B/HLLD-fix11 51e9c67
cd $B/HLLD-fix11
git apply $S/hr_hlld.diff
mkdir -p data
echo "rmhd: $(git -C $B/rmhd_final-loose10 rev-parse --short HEAD) + $(git -C $B/rmhd_final-loose10 diff --stat | tail -1) + coplanar7_b.py"
echo "HLLD: $(git -C $B/HLLD-fix11 rev-parse --short HEAD) + $(git -C $B/HLLD-fix11 diff --stat | tail -1)"
grep -c "RMHD_COPLANAR7_CALLS" $B/rmhd_final-loose10/rmhd/batched/coplanar7_b.py
grep -c "_HARVEST_RESCUES" $B/HLLD-fix11/src/physics/exact_flux.py
grep -c "_RESCUE_COLUMNS" $B/HLLD-fix11/src/physics/harvest.py
ls -la $B/rmhd_final-loose10/data/
