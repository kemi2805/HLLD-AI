#!/bin/bash
# Calea worktrees for the planar seven-wave stage (RMHD_COPLANAR7): rmhd 9ca5a4b +
# c7_rmhd.diff + coplanar7_b.py (final2 + coplanar rotation in planar4 + the stage
# in api.solve_batch), HLLD 51e9c67 + c7_hlld.diff (final2 + gate + rotation
# rescue + the ray for planar answers).  Uncommitted.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/rmhd_final
git worktree add --detach $B/rmhd_final-loose7 9ca5a4b
cd $B/rmhd_final-loose7
git apply $S/c7_rmhd.diff
cp $S/coplanar7_b.py rmhd/batched/coplanar7_b.py
mkdir -p data
for f in $B/rmhd_final-loose5/data/*; do cp -P "$f" data/; done
cd $B/HLLD
git worktree add --detach $B/HLLD-fix8 51e9c67
cd $B/HLLD-fix8
git apply $S/c7_hlld.diff
mkdir -p data
echo "rmhd: $(git -C $B/rmhd_final-loose7 rev-parse --short HEAD) + $(git -C $B/rmhd_final-loose7 diff --stat | tail -1) + coplanar7_b.py"
echo "HLLD: $(git -C $B/HLLD-fix8 rev-parse --short HEAD) + $(git -C $B/HLLD-fix8 diff --stat | tail -1)"
grep -c "CP7\|cop7" $B/rmhd_final-loose7/rmhd/batched/api.py
grep -c "coplanar7" $B/HLLD-fix8/src/physics/exact_flux.py
