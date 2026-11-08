#!/bin/bash
# rmhd_final-loose8: 9ca5a4b + c7a_rmhd.diff + coplanar7_b.py with the "after" mode
# (RMHD_COPLANAR7=after).  Uncommitted.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/rmhd_final
git worktree add --detach $B/rmhd_final-loose8 9ca5a4b
cd $B/rmhd_final-loose8
git apply $S/c7a_rmhd.diff
cp $S/coplanar7_b_after.py rmhd/batched/coplanar7_b.py
mkdir -p data
for f in $B/rmhd_final-loose5/data/*; do cp -P "$f" data/; done
echo "rmhd: $(git rev-parse --short HEAD) + $(git diff --stat | tail -1) + coplanar7_b.py"
grep -c "def after" rmhd/batched/coplanar7_b.py
