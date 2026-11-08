#!/bin/bash
# Calea worktrees for the planar seven-wave stage as a rung after the five-wave
# solver (RMHD_COPLANAR7=late): rmhd_final-loose9 = 9ca5a4b + c7l_rmhd.diff
# (= c7a_rmhd.diff) + coplanar7_b_late.py; HLLD-fix10 = 51e9c67 + c7l_hlld.diff
# (fix9's patch + `_coplanar7_and_ray` + the coverage flag).  Uncommitted.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/rmhd_final
git worktree add --detach $B/rmhd_final-loose9 9ca5a4b
cd $B/rmhd_final-loose9
git apply $S/c7l_rmhd.diff
cp $S/coplanar7_b_late.py rmhd/batched/coplanar7_b.py
cp $S/test_coplanar7.py tests/batched/test_coplanar7.py
cp $S/test_planar4_rotation.py tests/batched/test_planar4_rotation.py
mkdir -p data
for f in $B/rmhd_final-loose5/data/*; do cp -P "$f" data/; done
cd $B/HLLD
git worktree add --detach $B/HLLD-fix10 51e9c67
cd $B/HLLD-fix10
git apply $S/c7l_hlld.diff
mkdir -p data
echo "rmhd: $(git -C $B/rmhd_final-loose9 rev-parse --short HEAD) + $(git -C $B/rmhd_final-loose9 diff --stat | tail -1) + coplanar7_b.py"
echo "HLLD: $(git -C $B/HLLD-fix10 rev-parse --short HEAD) + $(git -C $B/HLLD-fix10 diff --stat | tail -1)"
grep -c "def late" $B/rmhd_final-loose9/rmhd/batched/coplanar7_b.py
grep -c "_coplanar7_and_ray\|_COPLANAR7_LATE" $B/HLLD-fix10/src/physics/exact_flux.py
