#!/bin/bash
# HLLD-fix9: 51e9c67 + c7_hlld.diff with the batched starts (RMHD_PLANAR4_STAR_BATCH).
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/HLLD
git worktree add --detach $B/HLLD-fix9 51e9c67
cd $B/HLLD-fix9
git apply $S/c7_hlld.diff
mkdir -p data
echo "HLLD: $(git rev-parse --short HEAD) + $(git diff --stat | tail -1)"
grep -c "_PLANAR4_STAR_BATCH" src/physics/exact_flux.py
