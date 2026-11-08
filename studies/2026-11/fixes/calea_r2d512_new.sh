#!/bin/bash
#SBATCH --job-name=r2d512-new
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=50-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The four-quadrant 512^2 run 77227 (ft3, production's chain of 11-03, every
# face harvested; results/r2d_512_exact_ml_r2d_it3) resumed from its step-1380
# restart with the NEW chain (user 2026-11-07: "yes cancel, and rerun, do we
# have checkpoints?"): the loss fixes, the admissibility gate, the rotation
# re-solve, the planar seven-wave rung after the five-wave solver, the batched
# starts, the complete harvest (rescue rows, reclassified lanes), and the
# production kernel set of rmhd_final-loose11 (scripts/kernels.env: the
# compiled planar root-finders RMHD_PLANAR_NJIT and the merged sweeps
# RMHD_MERGE_SWEEPS -- user 2026-11-07: "take those new kernels").  Network ft3
# and the two 25M retry networks, as in the original run.
#   sbatch --dependency=afterany:77227 calea_r2d512_new.sh
# 1. a 2-step pilot from the original run's step-1370 restart with every new
#    switch OFF: its diag rows against the original run's 1371-1372 (bitwise);
# 2. the run, from the step-1380 restart (copied from the original run on the
#    first start; a resubmission continues from its own).
# Worktrees rmhd_final-loose11 and HLLD-fix14 (c7_patches/setup_512.sh),
# uncommitted.  Output results/r2d_512_exact_ml_r2d_it3_newchain.
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
OLD=$R/r2d_512_exact_ml_r2d_it3
OUT=$R/r2d_512_exact_ml_r2d_it3_newchain
CK=$R/r2d_retrain/it3/ml_guess_r2d_it3.pt
export PYTHONPATH=$B/rmhd_final-loose11 HLLD_DIR=$B/HLLD-fix14
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPTS RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
unset RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS RMHD_PLANAR4_STAR_SET RMHD_PLANAR4_STAR_ITER
unset RMHD_ADMISSIBLE_GATE RMHD_ADMISSIBLE_GATE_DS RMHD_PLANAR4_ROT RMHD_COPLANAR7 RMHD_COPLANAR7_P RMHD_COPLANAR7_CALLS
unset RMHD_PLANAR4_STAR_BATCH RMHD_HARVEST_RESCUES RMHD_PLANAR_NJIT RMHD_MERGE_SWEEPS
export PROBLEM=riemann2d N=512 NSNAP=40 RMHD_ML_CKPT=$CK HARVEST_MAX=2000000000 HARVEST_FLUSH=200000
cd "$HLLD_DIR" || exit 1
grep -q _rendezvous src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
grep -q harvest-max scripts/run_2d.py || { echo "PREFLIGHT FAILED: harvest-max patch"; exit 2; }
grep -q RMHD_PLANAR_NJIT "$PYTHONPATH/scripts/kernels.env" || { echo "PREFLIGHT FAILED: kernels.env"; exit 2; }
[ -f "$CK" ] || { echo "PREFLIGHT FAILED: no checkpoint $CK"; exit 2; }
[ -f "$OUT/pilot_restart_1370.npz" ] || { echo "PREFLIGHT FAILED: no pilot restart"; exit 2; }
echo "== $(date) $(hostname): HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $PYTHONPATH rev-parse --short HEAD)+patch"

P=$OUT/pilot_off
if [ ! -f "$OUT/pilot_done" ]; then
    rm -rf "$P"; mkdir -p "$P"
    cp -p "$OUT/pilot_restart_1370.npz" "$P/restart.npz"
    RMHD_PLANAR_NJIT=0 RMHD_MERGE_SWEEPS=0 OUT=$P LOG=$B/HLLD/logs/r2d512new_pilot_$(date +%Y%m%d_%H%M%S).log \
        MAX_STEPS=2 RESUME=1 HARVEST_ALL=0 bash scripts/calea_rotor_exact.sh > /dev/null 2>&1
    want=$(grep -E "^137[12]," "$OLD/diag.csv")
    got=$(grep -E "^137[12]," "$P/diag.csv" 2>/dev/null)
    if [ -n "$got" ] && [ "$want" = "$got" ]; then
        echo "== PILOT (new switches off, from step 1370) MATCHES the original run's steps 1371-1372 bitwise"
        touch "$OUT/pilot_done"
    else
        echo "== PILOT DIFFERS from the original run:"
        diff <(echo "$want") <(echo "$got")
        exit 3
    fi
fi

[ -f "$OUT/restart.npz" ] || cp -p "$OLD/restart.npz" "$OUT/restart.npz"
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1
export RMHD_PLANAR4_ROT=1 RMHD_COPLANAR7=late RMHD_PLANAR4_STAR_BATCH=1 RMHD_HARVEST_RESCUES=1
echo "== $(date) resuming from step $(python3 -c "import numpy as np; print(int(np.load('$OUT/restart.npz')['step']))") with the new chain"
OUT=$OUT LOG=$B/HLLD/logs/r2d_512_exact_ml_r2d_it3_newchain_$(date +%Y%m%d_%H%M%S).log RESUME=1 \
    bash scripts/calea_rotor_exact.sh
echo "== $(date) exit $?"
