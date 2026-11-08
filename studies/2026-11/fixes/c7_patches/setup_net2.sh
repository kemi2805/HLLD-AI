#!/bin/bash
# Calea rmhd worktree with the planar network and the regular-answer
# preference ON BY DEFAULT (2026-11-08, user: "yes, turn the planar network on
# and prefer regular answers"): rmhd_final-loose13 = loose12's working tree
# (rsynced as it is) + net2_files/ (coplanar7_b.py with RMHD_COPLANAR7_NET
# defaulting to data/ml_planar_v2a.pt and RMHD_COPLANAR7_REGULAR on; the late
# rung's tests pinned to production's seed; test_planar_net.py).  loose11 (the
# 512^2 run) and loose12 (the 11-08 replay) are left alone.  Uncommitted.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
W=$B/rmhd_final-loose13
cd $B/rmhd_final
git worktree add --detach $W 9ca5a4b
rsync -a --exclude .git --exclude data --exclude __pycache__ --exclude '*.pyc' $B/rmhd_final-loose12/ $W/
cp -r $S/net2_files/rmhd $S/net2_files/tests $W/
mkdir -p $W/data
for f in $B/rmhd_final-loose12/data/*; do cp -P "$f" $W/data/; done
cd $W
echo "rmhd: $(git rev-parse --short HEAD) + $(git diff --stat | tail -1)"
md5sum rmhd/batched/coplanar7_b.py tests/batched/test_planar_net.py tests/batched/test_coplanar7.py \
    tests/batched/test_planar_njit.py data/ml_planar_v2a.pt
echo "-- differences from loose12 (expected: coplanar7_b.py and the three test files):"
diff -rq --exclude .git --exclude __pycache__ $B/rmhd_final-loose12 $W || true
