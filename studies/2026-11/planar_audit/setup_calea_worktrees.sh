#!/bin/bash
# Create the two detached worktrees for the admissibility replay and apply the
# uncommitted patches (git only; run on the login node).  Idempotent.
set -eu
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
if [ ! -d "$B/HLLD-admiss" ]; then
  git -C "$B/HLLD" worktree add --detach "$B/HLLD-admiss" 51e9c67
fi
if [ ! -d "$B/rmhd_final-admiss" ]; then
  git -C "$B/rmhd_final" worktree add --detach "$B/rmhd_final-admiss" 9ca5a4b
fi
cd "$B/HLLD-admiss" && git apply --check "$S/exact_flux_admissible.diff" 2>/dev/null && git apply "$S/exact_flux_admissible.diff" || echo "HLLD patch already applied?"
cd "$B/rmhd_final-admiss" && git apply --check "$S/slow_shock_admissible.diff" 2>/dev/null && git apply "$S/slow_shock_admissible.diff" || echo "rmhd patch already applied?"
mkdir -p "$B/rmhd_final-admiss/data"
for f in "$B"/rmhd_final/data/ml_guess_*.pt; do
  ln -sfn "$f" "$B/rmhd_final-admiss/data/$(basename "$f")"
done
git -C "$B/HLLD-admiss" diff --stat
git -C "$B/rmhd_final-admiss" diff --stat
ls "$B/rmhd_final-admiss/data" | head -20
