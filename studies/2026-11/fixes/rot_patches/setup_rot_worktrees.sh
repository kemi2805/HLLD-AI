#!/bin/bash
# Calea worktrees for the rotation re-solve (RMHD_PLANAR4_ROT): rmhd 9ca5a4b +
# rot_rmhd.diff (final2 + coplanar rotation in planar4), HLLD 51e9c67 +
# rot_hlld.diff (final2 + gate + rotation rescue).  Uncommitted, like loose5/fix6.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/rot_patches
cd $B/rmhd_final
git worktree add --detach $B/rmhd_final-loose6 9ca5a4b
cd $B/rmhd_final-loose6
git apply $S/rot_rmhd.diff
mkdir -p data
for f in $B/rmhd_final-loose5/data/*; do cp -P "$f" data/; done
cd $B/HLLD
git worktree add --detach $B/HLLD-fix7 51e9c67
cd $B/HLLD-fix7
git apply $S/rot_hlld.diff
mkdir -p data
echo "rmhd: $(git -C $B/rmhd_final-loose6 rev-parse --short HEAD) + $(git -C $B/rmhd_final-loose6 diff --stat | tail -1)"
echo "HLLD: $(git -C $B/HLLD-fix7 rev-parse --short HEAD) + $(git -C $B/HLLD-fix7 diff --stat | tail -1)"
grep -c "coplanar_rotation" $B/rmhd_final-loose6/rmhd/batched/fullcontact_b.py $B/rmhd_final-loose6/rmhd/batched/planar4_b.py
grep -c "_PLANAR4_ROT" $B/HLLD-fix7/src/physics/exact_flux.py
ls -la $B/rmhd_final-loose6/data
