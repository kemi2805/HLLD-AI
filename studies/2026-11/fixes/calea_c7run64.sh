#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired 64^2 run with the planar seven-wave stage (RMHD_COPLANAR7; user 2026-11-05: "give the seven-wave
# solver signed-field slow waves and the rotation. be ambitious") on top of the gate and the rotation re-solve
# (RMHD_PLANAR4_ROT; user 2026-11-05: "take a look on the
# entropy fall, we may get this also up to 99.99%"); the gate itself (RMHD_ADMISSIBLE_GATE; user
# 2026-11-05: "build the acceptance check and test it paired"). The gate refuses
# an exact answer that holds an entropy-falling jump (by more than 1e-9), a
# compressive fan, or waves out of order; a refused face goes on to the next
# rung, and to HLLD if none answers it.
#   MODE=finalrotc7  the final loss fixes + the rotation re-solve + the planar seven-wave stage
#               + the gate          pair: *_64_exact_finalrotgate (79440-79442) and the control
#   MAX_STEPS / SUFFIX (optional): a short run, written to *_64_exact_finalrotc7${SUFFIX}gate
#   (MODE=final / prod as in calea_gate64.sh)
#   sbatch --job-name=c7-r2d   --export=ALL,MODE=finalrotc7,PROBLEM=riemann2d,SHORT=r2d,CTRL=r2d_64_exact_ctl1102 calea_c7run64.sh
#   sbatch --job-name=c7-ot    --export=ALL,MODE=finalrotc7,PROBLEM=orszag_tang,SHORT=ot,CTRL=ot_64_exact_ssretry calea_c7run64.sh
#   sbatch --job-name=c7-rotor --export=ALL,MODE=finalrotc7,PROBLEM=rotor,SHORT=rotor,CTRL=rotor_64_exact_ssretry calea_c7run64.sh
# 1. a 4-step pilot from the patched worktrees with every switch OFF, its
#    diag.csv against the control's first rows (bitwise: no timing columns);
# 2. the full run with the MODE's switches on, production's settings otherwise
#    (calea_rotor_exact.sh defaults, pool 32x2, harvest-all).
# Worktrees HLLD-fix8 (51e9c67) and rmhd_final-loose7 (9ca5a4b), patches
# uncommitted: fixes/c7_patches.
set -u
: "${PROBLEM:?}" "${SHORT:?}" "${CTRL:?}" "${MODE:?}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
export PYTHONPATH=$B/rmhd_final-loose7 HLLD_DIR=$B/HLLD-fix8
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
unset RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS RMHD_PLANAR4_STAR_SET RMHD_PLANAR4_STAR_ITER RMHD_ADMISSIBLE_GATE RMHD_ADMISSIBLE_GATE_DS RMHD_PLANAR4_ROT RMHD_COPLANAR7
export PROBLEM N=64
cd "$HLLD_DIR" || exit 1
grep -q coplanar7 src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
test -f "$PYTHONPATH/rmhd/batched/coplanar7_b.py" || { echo "PREFLIGHT FAILED: rmhd patch"; exit 2; }
echo "== $(date) $(hostname) $PROBLEM ($MODE): HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $PYTHONPATH rev-parse --short HEAD)+patch; control $CTRL"
P=$R/fixes/pilots_c7/${SHORT}_${MODE}_64_off
rm -rf "$P"
mkdir -p "$R/fixes/pilots_c7"
OUT=$P LOG=$B/HLLD/logs/fixc7_pilot_${SHORT}_${MODE}_$(date +%Y%m%d_%H%M%S).log MAX_STEPS=4 RESUME=0 HARVEST_ALL=0 \
    bash scripts/calea_rotor_exact.sh > /dev/null 2>&1
if [ -f "$P/diag.csv" ] && diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv") > /dev/null; then
    echo "== PILOT (switches off) MATCHES $CTRL: diag rows 1-4 bitwise"
else
    echo "== PILOT (switches off) DIFFERS from $CTRL:"
    diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv" 2>/dev/null)
fi
case "$MODE" in final|finalrot|finalrotc7) export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 ;; esac
case "$MODE" in finalrot|finalrotc7) export RMHD_PLANAR4_ROT=1 ;; esac
[ "$MODE" = finalrotc7 ] && export RMHD_COPLANAR7=1
export RMHD_ADMISSIBLE_GATE=1
echo "== $(date) treatment ($MODE): LOOSE_RETRY=${RMHD_SLOWSHOCK_LOOSE_RETRY:-0} PLANAR4_STAR=${RMHD_PLANAR4_STAR:-0} ADMISSIBLE_GATE=$RMHD_ADMISSIBLE_GATE PLANAR4_ROT=${RMHD_PLANAR4_ROT:-0} COPLANAR7=${RMHD_COPLANAR7:-0} MAX_STEPS=${MAX_STEPS:-all}"
TAG=_${MODE}${SUFFIX:-}gate OUT=$R/${SHORT}_64_exact_${MODE}${SUFFIX:-}gate LOG=$B/HLLD/logs/${SHORT}_64_exact_${MODE}${SUFFIX:-}gate_$(date +%Y%m%d_%H%M%S).log \
    bash scripts/calea_rotor_exact.sh
echo "== $(date) exit $?"
