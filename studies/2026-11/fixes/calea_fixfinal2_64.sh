#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired 64^2 run with the FINAL loss-fix configuration of 2026-11-05 (user: "try them out
# and lets see if we can get to 100%"):
#   RMHD_SLOWSHOCK_LOOSE_RETRY=1  the slow-shock lanes answered by the ladder's
#                                 loose fallback are offered to the retry
#   RMHD_PLANAR4_STAR=1           the four-unknown rung's lost lanes again from
#                                 HLLD's star pressure (pressure/field steering)
#   sbatch --job-name=fix64-r2d   --export=ALL,PROBLEM=riemann2d,SHORT=r2d,CTRL=r2d_64_exact_ctl1102 calea_fix64.sh
#   sbatch --job-name=fix64-ot    --export=ALL,PROBLEM=orszag_tang,SHORT=ot,CTRL=ot_64_exact_ssretry calea_fix64.sh
#   sbatch --job-name=fix64-rotor --export=ALL,PROBLEM=rotor,SHORT=rotor,CTRL=rotor_64_exact_ssretry calea_fix64.sh
# 1. a 4-step pilot from the patched worktrees with both switches OFF, its
#    diag.csv against the control's first rows (bitwise: no timing columns);
# 2. the full run with both switches on, production's settings otherwise
#    (calea_rotor_exact.sh defaults, pool 32x2, harvest-all).
# Worktrees HLLD-fix5 (51e9c67) and rmhd_final-loose5 (9ca5a4b), patches
# uncommitted: fixes/final2_patches/final2_{hlld,rmhd}.diff (lean starts, signed-field slow wave, near-side and both-sides fast starts; entropy may fall by 1e-4).
set -u
: "${PROBLEM:?}" "${SHORT:?}" "${CTRL:?}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
export PYTHONPATH=$B/rmhd_final-loose5 HLLD_DIR=$B/HLLD-fix5
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
unset RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS RMHD_PLANAR4_STAR_SET RMHD_PLANAR4_STAR_ITER
export PROBLEM N=64
cd "$HLLD_DIR" || exit 1
grep -q _PLANAR4_STAR_SIDE src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
grep -q LOOSE_RETRY "$PYTHONPATH/rmhd/batched/slow_shock_b.py" || { echo "PREFLIGHT FAILED: rmhd patch"; exit 2; }
echo "== $(date) $(hostname) $PROBLEM: HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $PYTHONPATH rev-parse --short HEAD)+patch; control $CTRL"
P=$R/fixes/pilots_final2/${SHORT}_64_off
rm -rf "$P"
mkdir -p "$R/fixes/pilots_final2"
OUT=$P LOG=$B/HLLD/logs/fixfinal2_pilot_${SHORT}_$(date +%Y%m%d_%H%M%S).log MAX_STEPS=4 RESUME=0 HARVEST_ALL=0 \
    bash scripts/calea_rotor_exact.sh > /dev/null 2>&1
if [ -f "$P/diag.csv" ] && diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv") > /dev/null; then
    echo "== PILOT (switches off) MATCHES $CTRL: diag rows 1-4 bitwise"
else
    echo "== PILOT (switches off) DIFFERS from $CTRL:"
    diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv" 2>/dev/null)
fi
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4
echo "== $(date) treatment: RMHD_SLOWSHOCK_LOOSE_RETRY=$RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR=$RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS=$RMHD_PLANAR4_STAR_DS (lean starts)"
TAG=_fixfinal2 OUT=$R/${SHORT}_64_exact_fixfinal2 LOG=$B/HLLD/logs/${SHORT}_64_exact_fixfinal2_$(date +%Y%m%d_%H%M%S).log \
    bash scripts/calea_rotor_exact.sh
echo "== $(date) exit $?"
