#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=50-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The resolution ladder (user 2026-11-08: "Now with the fixes it is time to
# build the resolution ladder. Run 4Q, rotor and OT with 128, 256, 512 (except
# 4Q). The higher resolution only after the lower finished"): one run of
# PROBLEM (riemann2d | rotor | orszag_tang) at N, with everything of 11-07/08:
#   the new chain   loss fixes, admissibility gate, rotation re-solve, the late
#                   planar seven-wave rung, batched starts
#   the kernels     scripts/kernels.env: RMHD_PLANAR_NJIT, RMHD_MERGE_SWEEPS
#   the late rung   the planar network and regular answers first (on by
#                   default in rmhd_final-loose13)
#   the harvest     every face, with the rescue rows (RMHD_HARVEST_RESCUES),
#                   200k-row shards, no cap -- coverage maps for every step
# Networks: the 4Q with ft3 (as its 512^2 run), the rotor and OT with
# production's rotor_ft; the two 25M retry networks for all.  NSNAP 40.
# Worktrees rmhd_final-loose13 + HLLD-fix14, uncommitted.
#   sbatch --job-name=lad-rotor-128 --export=ALL,PROBLEM=rotor,N=128 calea_ladder.sh
# -> results/<short>_<N>_exact_ladder
set -u
: "${PROBLEM:?}" "${N:?}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
export PYTHONPATH=$B/rmhd_final-loose13 HLLD_DIR=$B/HLLD-fix14
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
unset RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS RMHD_PLANAR4_STAR_SET RMHD_PLANAR4_STAR_ITER
unset RMHD_ADMISSIBLE_GATE RMHD_ADMISSIBLE_GATE_DS RMHD_PLANAR4_ROT RMHD_COPLANAR7 RMHD_COPLANAR7_P RMHD_COPLANAR7_CALLS
unset RMHD_PLANAR4_STAR_BATCH RMHD_HARVEST_RESCUES RMHD_PLANAR_NJIT RMHD_MERGE_SWEEPS
unset RMHD_COPLANAR7_NET RMHD_COPLANAR7_NET_FALLBACK RMHD_COPLANAR7_REGULAR
if [ "$PROBLEM" = riemann2d ]; then
    export RMHD_ML_CKPT=$R/r2d_retrain/it3/ml_guess_r2d_it3.pt
    [ -f "$RMHD_ML_CKPT" ] || { echo "PREFLIGHT FAILED: no checkpoint $RMHD_ML_CKPT"; exit 2; }
fi
case "$PROBLEM" in riemann2d) SHORT=r2d ;; rotor) SHORT=rotor ;; orszag_tang) SHORT=ot ;; *) echo "unknown PROBLEM"; exit 2 ;; esac
export PROBLEM N NSNAP=40 OUT=$R/${SHORT}_${N}_exact_ladder${PILOT_TAG:-} HARVEST_MAX=2000000000 HARVEST_FLUSH=200000
# pilots: PILOT_STEPS (MAX_STEPS), PILOT_TAG (an OUT suffix), LATE_NET / LATE_REGULAR (the late rung's switches)
[ -n "${PILOT_STEPS:-}" ] && export MAX_STEPS=$PILOT_STEPS
[ -n "${LATE_NET:-}" ] && export RMHD_COPLANAR7_NET=$LATE_NET
[ -n "${LATE_REGULAR:-}" ] && export RMHD_COPLANAR7_REGULAR=$LATE_REGULAR
cd "$HLLD_DIR" || exit 1
grep -q _rendezvous src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
grep -q harvest-max scripts/run_2d.py || { echo "PREFLIGHT FAILED: harvest-max patch"; exit 2; }
grep -q RMHD_PLANAR_NJIT "$PYTHONPATH/scripts/kernels.env" || { echo "PREFLIGHT FAILED: kernels.env"; exit 2; }
grep -q RMHD_COPLANAR7_REGULAR "$PYTHONPATH/rmhd/batched/coplanar7_b.py" || { echo "PREFLIGHT FAILED: rmhd is not loose13"; exit 2; }
[ -f "$PYTHONPATH/data/ml_planar_v2a.pt" ] || { echo "PREFLIGHT FAILED: no planar network"; exit 2; }
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1
export RMHD_PLANAR4_ROT=1 RMHD_COPLANAR7=late RMHD_PLANAR4_STAR_BATCH=1 RMHD_HARVEST_RESCUES=1
"$HOME/venv/rmhd/bin/python" -c "import rmhd.batched.coplanar7_b as C; print('late rung: network %r, fallback %s, regular %s' % (C.NET, C.NET_FALLBACK, C.REGULAR))"
echo "== $(date) $(hostname): $PROBLEM $N, HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $PYTHONPATH rev-parse --short HEAD)+patch"
LOG=$B/HLLD/logs/ladder_${PROBLEM}_${N}_$(date +%Y%m%d_%H%M%S).log RESUME=1 bash scripts/calea_rotor_exact.sh
st=$?
echo "== $(date) exit $st"
exit $st
