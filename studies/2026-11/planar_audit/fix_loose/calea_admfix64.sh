#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired 64^2 rerun of the slow-shock admissibility filter WITH the loose-
# fallback fix (user, 2026-11-04: "Yes, fix and rerun"; the first filter runs
# 77965-77967 accepted the ladder's grid cell, no root, wherever the filter had
# discarded every root).
#   sbatch --job-name=admfix-r2d   --export=ALL,PROBLEM=riemann2d,SHORT=r2d,CTRL=r2d_64_exact_ctl1102 calea_admfix64.sh
#   sbatch --job-name=admfix-ot    --export=ALL,PROBLEM=orszag_tang,SHORT=ot,CTRL=ot_64_exact_ssretry calea_admfix64.sh
#   sbatch --job-name=admfix-rotor --export=ALL,PROBLEM=rotor,SHORT=rotor,CTRL=rotor_64_exact_ssretry calea_admfix64.sh
# 1. a 4-step pilot from the patched worktrees with every new switch OFF, its
#    diag.csv against the control's first rows (bitwise: no timing columns);
# 2. the full run with RMHD_SLOWSHOCK_ADMISSIBLE=1 and RMHD_ADMISSIBLE=report
#    (count, never refuse), production's settings otherwise (calea_rotor_exact.sh
#    defaults, pool 32x2, harvest-all) -- exactly the first round's treatment.
# Worktrees HLLD-admiss (51e9c67 + exact_flux_admissible.diff) and
# rmhd_final-admiss (9ca5a4b + slow_shock_admissible_1104.diff), uncommitted.
set -u
: "${PROBLEM:?}" "${SHORT:?}" "${CTRL:?}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
export PYTHONPATH=$B/rmhd_final-admiss HLLD_DIR=$B/HLLD-admiss
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
export PROBLEM N=64
cd "$HLLD_DIR" || exit 1
echo "== $(date) $(hostname) $PROBLEM: HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $B/rmhd_final-admiss rev-parse --short HEAD)+patch ($(grep -c discarded $B/rmhd_final-admiss/rmhd/batched/slow_shock_b.py) 'discarded' lines); control $CTRL"
P=$R/r2d_retrain/planar_audit/pilots/${SHORT}_64_off_fix
rm -rf "$P"
OUT=$P LOG=$B/HLLD/logs/admfix64_pilot_${SHORT}_$(date +%Y%m%d_%H%M%S).log MAX_STEPS=4 RESUME=0 HARVEST_ALL=0 \
    bash scripts/calea_rotor_exact.sh > /dev/null 2>&1
if [ -f "$P/diag.csv" ] && diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv") > /dev/null; then
    echo "== PILOT (switches off) MATCHES $CTRL: diag rows 1-4 bitwise"
else
    echo "== PILOT (switches off) DIFFERS from $CTRL:"
    diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv" 2>/dev/null)
fi
export RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=report
echo "== $(date) treatment: RMHD_SLOWSHOCK_ADMISSIBLE=$RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE=$RMHD_ADMISSIBLE"
TAG=_admfix OUT=$R/${SHORT}_64_exact_admfix LOG=$B/HLLD/logs/${SHORT}_64_exact_admfix_$(date +%Y%m%d_%H%M%S).log \
    bash scripts/calea_rotor_exact.sh
echo "== $(date) exit $?"
