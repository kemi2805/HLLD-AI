#!/bin/bash
# Calea HLLD worktree for the 4Q 512^2 run resumed with the new chain (user
# 2026-11-07: "yes cancel, and rerun, do we have checkpoints?"): HLLD-fix14 =
# HLLD-fix13 (51e9c67 + lev2_hlld.diff + lev2_files: the late rung, the rescue
# harvest, the merged sweeps) + patch_harvest_max.py (--harvest-max: the 512^2
# harvest keeps every row, as the original run 77227 did).  Uncommitted.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/HLLD
git worktree add --detach $B/HLLD-fix14 51e9c67
cd $B/HLLD-fix14
git apply $S/lev2_hlld.diff
cp -r $S/lev2_files/src $S/lev2_files/tests .
python3 /mnt/rafast/miler/scratch/r2d_retrain/patch_harvest_max.py $B/HLLD-fix14
mkdir -p data
echo "HLLD: $(git rev-parse --short HEAD) + $(git diff --stat | tail -1)"
grep -c "_rendezvous" src/physics/exact_flux.py
grep -c "harvest-max" scripts/run_2d.py
