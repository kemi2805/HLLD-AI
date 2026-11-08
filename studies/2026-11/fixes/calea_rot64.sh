#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired 64^2 run with the gate AND the rotation re-solve (RMHD_PLANAR4_ROT; user 2026-11-05: "take a look on the
# entropy fall, we may get this also up to 99.99%"); the gate itself (RMHD_ADMISSIBLE_GATE; user
# 2026-11-05: "build the acceptance check and test it paired"). The gate refuses
# an exact answer that holds an entropy-falling jump (by more than 1e-9), a
# compressive fan, or waves out of order; a refused face goes on to the next
# rung, and to HLLD if none answers it.
#   MODE=finalrot  the final loss fixes (loose retry, planar4 star starts, DS 1e-4) + the rotation re-solve
#               + the gate          pair: *_64_exact_finalgate, *_64_exact_fixfinal2 and the control
#   (MODE=final / prod as in calea_gate64.sh)
#   sbatch --job-name=rot-fin-r2d   --export=ALL,MODE=finalrot,PROBLEM=riemann2d,SHORT=r2d,CTRL=r2d_64_exact_ctl1102 calea_rot64.sh
#   sbatch --job-name=rot-fin-ot    --export=ALL,MODE=finalrot,PROBLEM=orszag_tang,SHORT=ot,CTRL=ot_64_exact_ssretry calea_rot64.sh
#   sbatch --job-name=rot-fin-rotor --export=ALL,MODE=finalrot,PROBLEM=rotor,SHORT=rotor,CTRL=rotor_64_exact_ssretry calea_rot64.sh
# 1. a 4-step pilot from the patched worktrees with every switch OFF, its
#    diag.csv against the control's first rows (bitwise: no timing columns);
# 2. the full run with the MODE's switches on, production's settings otherwise
#    (calea_rotor_exact.sh defaults, pool 32x2, harvest-all).
# Worktrees HLLD-fix7 (51e9c67) and rmhd_final-loose6 (9ca5a4b), patches
# uncommitted: fixes/rot_patches/{rot_hlld,rot_rmhd}.diff.
set -u
: "${PROBLEM:?}" "${SHORT:?}" "${CTRL:?}" "${MODE:?}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
export PYTHONPATH=$B/rmhd_final-loose6 HLLD_DIR=$B/HLLD-fix7
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
unset RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS RMHD_PLANAR4_STAR_SET RMHD_PLANAR4_STAR_ITER RMHD_ADMISSIBLE_GATE RMHD_ADMISSIBLE_GATE_DS RMHD_PLANAR4_ROT
export PROBLEM N=64
cd "$HLLD_DIR" || exit 1
grep -q _PLANAR4_ROT src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
grep -q coplanar_rotation "$PYTHONPATH/rmhd/batched/planar4_b.py" || { echo "PREFLIGHT FAILED: rmhd patch"; exit 2; }
echo "== $(date) $(hostname) $PROBLEM ($MODE): HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $PYTHONPATH rev-parse --short HEAD)+patch; control $CTRL"
P=$R/fixes/pilots_rot/${SHORT}_${MODE}_64_off
rm -rf "$P"
mkdir -p "$R/fixes/pilots_rot"
OUT=$P LOG=$B/HLLD/logs/fixrot_pilot_${SHORT}_${MODE}_$(date +%Y%m%d_%H%M%S).log MAX_STEPS=4 RESUME=0 HARVEST_ALL=0 \
    bash scripts/calea_rotor_exact.sh > /dev/null 2>&1
if [ -f "$P/diag.csv" ] && diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv") > /dev/null; then
    echo "== PILOT (switches off) MATCHES $CTRL: diag rows 1-4 bitwise"
else
    echo "== PILOT (switches off) DIFFERS from $CTRL:"
    diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv" 2>/dev/null)
fi
case "$MODE" in final|finalrot) export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 ;; esac
[ "$MODE" = finalrot ] && export RMHD_PLANAR4_ROT=1
export RMHD_ADMISSIBLE_GATE=1
echo "== $(date) treatment ($MODE): LOOSE_RETRY=${RMHD_SLOWSHOCK_LOOSE_RETRY:-0} PLANAR4_STAR=${RMHD_PLANAR4_STAR:-0} ADMISSIBLE_GATE=$RMHD_ADMISSIBLE_GATE PLANAR4_ROT=${RMHD_PLANAR4_ROT:-0}"
TAG=_${MODE}gate OUT=$R/${SHORT}_64_exact_${MODE}gate LOG=$B/HLLD/logs/${SHORT}_64_exact_${MODE}gate_$(date +%Y%m%d_%H%M%S).log \
    bash scripts/calea_rotor_exact.sh
echo "== $(date) exit $?"
