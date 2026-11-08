#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Harvest runs for the planar warm-start network (user 2026-11-06: "yes, launch
# the harvest runs when the late runs finish"; "make new ml models for the
# planar solver"): the late runs' configuration (81297-81299, tag
# finalrotc7lbgate) with the planar seven-wave stages in one solve
# (RMHD_COPLANAR7_CALLS=1: the same answers) and every rung's answers harvested
# (RMHD_HARVEST_RESCUES=1: source 2 the late rung, 3 the four-unknown solver,
# each row with its steering, rotations and stage; the lanes whose degenerate
# answer is refused after the harvest as unsolved rows, reason 4 -- 18.9k of
# the 4Q 64^2 run's rescue lanes were written nowhere before).
#   MODE=finalrotc7lh   loss fixes + gate + rotation re-solve + late rung (staged, the
#                       default) + batched starts + the rescue harvest
#   (MODE=finalrotc7l1h, the late rung in one call, is SLOWER: timing arms 81483/
#   81484, 11-06 -- the same answers bitwise, the rung 7-35% dearer; not used)
#   sbatch --job-name=hv-r2d --dependency=afterany:81299 --export=ALL,MODE=finalrotc7lh,PROBLEM=riemann2d,SHORT=r2d,CTRL=r2d_64_exact_ctl1102 calea_harvest64.sh
# 1. a 4-step pilot with every switch OFF, its diag.csv against the control's
#    first rows (bitwise: no timing columns);
# 2. the full run (calea_rotor_exact.sh defaults: pool 32x2, harvest-all).
# Its diag.csv should match the late run's (*_64_exact_finalrotc7lbgate) row
# for row: neither switch changes an answer.
# Worktrees HLLD-fix12 (51e9c67 + hr2_hlld.diff) and rmhd_final-loose10
# (9ca5a4b + c7l_rmhd.diff + coplanar7_b_onecall.py), uncommitted: fixes/c7_patches.
set -u
: "${PROBLEM:?}" "${SHORT:?}" "${CTRL:?}" "${MODE:?}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
export PYTHONPATH=$B/rmhd_final-loose10 HLLD_DIR=$B/HLLD-fix12
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
unset RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS RMHD_PLANAR4_STAR_SET RMHD_PLANAR4_STAR_ITER RMHD_ADMISSIBLE_GATE RMHD_ADMISSIBLE_GATE_DS RMHD_PLANAR4_ROT RMHD_COPLANAR7 RMHD_COPLANAR7_P RMHD_PLANAR4_STAR_BATCH RMHD_COPLANAR7_CALLS RMHD_HARVEST_RESCUES
export PROBLEM N=64
cd "$HLLD_DIR" || exit 1
grep -q R_RECLASSIFIED src/physics/harvest.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
grep -q RMHD_COPLANAR7_CALLS "$PYTHONPATH/rmhd/batched/coplanar7_b.py" || { echo "PREFLIGHT FAILED: rmhd patch"; exit 2; }
echo "== $(date) $(hostname) $PROBLEM ($MODE): HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $PYTHONPATH rev-parse --short HEAD)+patch; control $CTRL"
P=$R/fixes/pilots_hv/${SHORT}_${MODE}_64_off
rm -rf "$P"
mkdir -p "$R/fixes/pilots_hv"
OUT=$P LOG=$B/HLLD/logs/hv_pilot_${SHORT}_${MODE}_$(date +%Y%m%d_%H%M%S).log MAX_STEPS=4 RESUME=0 HARVEST_ALL=0 \
    bash scripts/calea_rotor_exact.sh > /dev/null 2>&1
if [ -f "$P/diag.csv" ] && diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv") > /dev/null; then
    echo "== PILOT (switches off) MATCHES $CTRL: diag rows 1-4 bitwise"
else
    echo "== PILOT (switches off) DIFFERS from $CTRL:"
    diff <(head -5 "$R/$CTRL/diag.csv") <(head -5 "$P/diag.csv" 2>/dev/null)
fi
case "$MODE" in
    finalrotc7l1h) export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_PLANAR4_ROT=1 \
                          RMHD_PLANAR4_STAR_BATCH=1 RMHD_COPLANAR7=late RMHD_COPLANAR7_CALLS=1 RMHD_HARVEST_RESCUES=1 ;;
    finalrotc7lh) export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_PLANAR4_ROT=1 \
                          RMHD_PLANAR4_STAR_BATCH=1 RMHD_COPLANAR7=late RMHD_HARVEST_RESCUES=1 ;;
    *) echo "unknown MODE $MODE"; exit 2 ;;
esac
export RMHD_ADMISSIBLE_GATE=1
echo "== $(date) treatment ($MODE): LOOSE_RETRY=$RMHD_SLOWSHOCK_LOOSE_RETRY PLANAR4_STAR=$RMHD_PLANAR4_STAR ADMISSIBLE_GATE=$RMHD_ADMISSIBLE_GATE PLANAR4_ROT=$RMHD_PLANAR4_ROT COPLANAR7=$RMHD_COPLANAR7 CALLS=${RMHD_COPLANAR7_CALLS:-4} STAR_BATCH=$RMHD_PLANAR4_STAR_BATCH HARVEST_RESCUES=$RMHD_HARVEST_RESCUES MAX_STEPS=${MAX_STEPS:-all}"
TAG=_${MODE}gate OUT=$R/${SHORT}_64_exact_${MODE}gate LOG=$B/HLLD/logs/${SHORT}_64_exact_${MODE}gate_$(date +%Y%m%d_%H%M%S).log \
    bash scripts/calea_rotor_exact.sh
echo "== $(date) exit $?"
