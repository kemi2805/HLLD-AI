#!/bin/bash
# Calea rmhd worktree for the planar network in the late rung (2026-11-08):
# rmhd_final-loose12 = loose11's working tree (9ca5a4b + the lever patch set,
# rsynced as it is) + net_files/ (coplanar7_b.py with RMHD_COPLANAR7_NET,
# planar_net.py, tests/batched/test_planar_net.py) + data/ml_planar_v2a.pt
# (results/planar_ml/train_v2).  loose11 is left alone: the 512^2 run 84516
# imports it.  Uncommitted.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
W=$B/rmhd_final-loose12
cd $B/rmhd_final
git worktree add --detach $W 9ca5a4b
rsync -a --exclude .git --exclude data --exclude __pycache__ --exclude '*.pyc' $B/rmhd_final-loose11/ $W/
cp -r $S/net_files/rmhd $S/net_files/tests $W/
mkdir -p $W/data
for f in $B/rmhd_final-loose11/data/*; do cp -P "$f" $W/data/; done
cp $B/HLLD/results/planar_ml/train_v2/planar_v2a.pt $W/data/ml_planar_v2a.pt
cd $W
echo "rmhd: $(git rev-parse --short HEAD) + $(git diff --stat | tail -1)"
md5sum rmhd/batched/coplanar7_b.py rmhd/batched/planar_net.py tests/batched/test_planar_net.py data/ml_planar_v2a.pt
echo "-- differences from loose11 (expected: coplanar7_b.py, planar_net.py, test_planar_net.py, the checkpoint):"
diff -rq --exclude .git --exclude __pycache__ $B/rmhd_final-loose11 $W || true
