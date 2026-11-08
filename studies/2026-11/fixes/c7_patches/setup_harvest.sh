#!/bin/bash
# Calea HLLD worktree for the harvest runs (user 2026-11-06: "yes, launch the
# harvest runs when the late runs finish"): HLLD-fix12 = 51e9c67 + hr2_hlld.diff
# (fix11's rescue rows + the reclassified lanes as unsolved rows, both under
# RMHD_HARVEST_RESCUES) + tests/test_harvest_rescues.py.  rmhd stays
# rmhd_final-loose10 (the one-call rung).  Uncommitted.  fix11 is left as it
# is: the dataset jobs 81485-81488 run on it.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/c7_patches
cd $B/HLLD
git worktree add --detach $B/HLLD-fix12 51e9c67
cd $B/HLLD-fix12
git apply $S/hr2_hlld.diff
cp $S/test_harvest_rescues.py tests/test_harvest_rescues.py
mkdir -p data
echo "HLLD: $(git -C $B/HLLD-fix12 rev-parse --short HEAD) + $(git -C $B/HLLD-fix12 diff --stat | tail -1)"
grep -c "_HARVEST_RESCUES" $B/HLLD-fix12/src/physics/exact_flux.py
grep -c "R_RECLASSIFIED" $B/HLLD-fix12/src/physics/harvest.py $B/HLLD-fix12/src/physics/exact_flux.py
