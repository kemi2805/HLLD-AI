#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired 64^2 run with the slow-shock admissibility filter (user, 2026-11-03:
# "Port the slow-shock root filter to njit and run paired 64^2 tests").
#   sbatch --job-name=adm64-r2d   --export=ALL,PROBLEM=riemann2d,SHORT=r2d,CTRL=r2d_64_exact_ctl1102 calea_adm64.sh
#   sbatch --job-name=adm64-ot    --export=ALL,PROBLEM=orszag_tang,SHORT=ot,CTRL=ot_64_exact_ssretry calea_adm64.sh
#   sbatch --job-name=adm64-rotor --export=ALL,PROBLEM=rotor,SHORT=rotor,CTRL=rotor_64_exact_ssretry calea_adm64.sh
# 1. a 4-step pilot from the patched worktrees with every new switch OFF, its
#    diag.csv against the control's first rows (bitwise: no timing columns);
# 2. the full run with RMHD_SLOWSHOCK_ADMISSIBLE=1 (admissible slow-shock roots
#    only) and RMHD_ADMISSIBLE=report (count, never refuse), production's
#    settings otherwise (calea_rotor_exact.sh defaults, pool 32x2, harvest-all).
# Worktrees HLLD-admiss (51e9c67) and rmhd_final-admiss (9ca5a4b), patches
# uncommitted: planar_audit/{exact_flux,slow_shock}_admissible.diff.
set -u
: "${PROBLEM:?}" "${SHORT:?}" "${CTRL:?}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
export PYTHONPATH=$B/rmhd_final-admiss HLLD_DIR=$B/HLLD-admiss
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
export PROBLEM N=64
cd "$HLLD_DIR" || exit 1
echo "== $(date) $(hostname) $PROBLEM: HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $B/rmhd_final-admiss rev-parse --short HEAD)+patch; control $CTRL"
P=$R/r2d_retrain/planar_audit/pilots/${SHORT}_64_off
rm -rf "$P"
OUT=$P LOG=$B/HLLD/logs/adm64_pilot_${SHORT}_$(date +%Y%m%d_%H%M%S).log MAX_STEPS=4 RESUME=0 HARVEST_ALL=0 \
    bash scripts/calea_rotor_exact.sh > /dev/null 2>&1
if [ -f "$P/diag.csv" ] && diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv") > /dev/null; then
    echo "== PILOT (switches off) MATCHES $CTRL: diag rows 1-4 bitwise"
else
    echo "== PILOT (switches off) DIFFERS from $CTRL:"
    diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv" 2>/dev/null)
fi
export RMHD_SLOWSHOCK_ADMISSIBLE=1 RMHD_ADMISSIBLE=report
echo "== $(date) treatment: RMHD_SLOWSHOCK_ADMISSIBLE=$RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE=$RMHD_ADMISSIBLE"
TAG=_admiss OUT=$R/${SHORT}_64_exact_admiss LOG=$B/HLLD/logs/${SHORT}_64_exact_admiss_$(date +%Y%m%d_%H%M%S).log \
    bash scripts/calea_rotor_exact.sh
echo "== $(date) exit $?"
