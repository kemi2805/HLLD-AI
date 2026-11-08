#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Lever 2 timing (user 2026-11-07): calea_cost_arms5.sh on HLLD-fix13 (c7_patches/setup_lever2.sh),
# adding the arm latebnjm = latebnj + RMHD_MERGE_SWEEPS=1 (one pipeline call per RK stage).
# Lever 1 timing (user 2026-11-07): calea_cost_arms4.sh on rmhd_final-loose11 / HLLD-fix12
# (c7_patches/setup_lever.sh), arms lateb (numpy), latebnj (RMHD_PLANAR_NJIT=1, the compiled
# root-finders) and latebck (=check: both, numpy answers, mismatches in njit_check.json).
# Earlier: calea_cost_arms3.sh on rmhd_final-loose10 / HLLD-fix11 (c7_patches/setup_onecall.sh), adding the
# arms lateb1 / lateb2: lateb with the planar seven-wave stages in one / two solves
# (RMHD_COPLANAR7_CALLS; the same answers as staged).  calea_cost_arms2.sh with RUNNER (default run_2d_timed.py; run_2d_stats.py adds the per-lane
# statistics of tpatch2.py -> $OUT/rung_stats.npz).  OT / rotor: is the late runs' gain over the rotation runs
# seven-wave rung or the batched starts (on only in the late runs)?  STEPS
# steps from ONE state -- a copy of the late run's restart -- under each of
# ARMS in turn on this node, the run's settings otherwise (calea_rotor_exact.sh),
# wall per stage from tpatch.py (arms as in calea_cost_arms.sh):
#   sbatch --job-name=cost-ot --export=ALL,PROBLEM=orszag_tang,SHORT=ot,RST=restart_ot_340.npz,STEPS=10,ARMS="rotb lateb" calea_cost_arms2.sh
set -u
: "${PROBLEM:?}" "${SHORT:?}" "${RST:?}" "${STEPS:?}" "${ARMS:?}"
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/cost_arms
RMHD=$B/rmhd_final-loose11
export PYTHONPATH=$RMHD HLLD_DIR=$B/HLLD-fix13
PY=$HOME/venv/rmhd/bin/python
case "$PROBLEM" in orszag_tang) TEND=1.0 ;; *) TEND=0.4 ;; esac
cd "$HLLD_DIR" || exit 1
grep -q _rendezvous src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
grep -q RMHD_PLANAR_NJIT "$RMHD/rmhd/batched/planar_njit.py" || { echo "PREFLIGHT FAILED: rmhd patch"; exit 2; }
[ -f "$S/$RST" ] || { echo "PREFLIGHT FAILED: no restart copy $RST"; exit 2; }
for ARM in $ARMS; do
    OUT=$B/HLLD/results/fixes/cost_arms/${SHORT}_$ARM
    unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE
    unset RMHD_MERGE_SWEEPS RMHD_PLANAR_NJIT RMHD_COPLANAR7_CALLS RMHD_HARVEST_RESCUES RMHD_SLOWSHOCK_LOOSE_RETRY RMHD_PLANAR4_STAR RMHD_PLANAR4_STAR_DS RMHD_PLANAR4_STAR_SET RMHD_PLANAR4_STAR_ITER RMHD_ADMISSIBLE_GATE RMHD_ADMISSIBLE_GATE_DS RMHD_PLANAR4_ROT RMHD_COPLANAR7 RMHD_COPLANAR7_P RMHD_PLANAR4_STAR_BATCH
    export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_PLANAR4_ROT=1 RMHD_ADMISSIBLE_GATE=1
    case "$ARM" in
        rot) ;;
        rotb) export RMHD_PLANAR4_STAR_BATCH=1 ;;
        lateb) export RMHD_PLANAR4_STAR_BATCH=1 RMHD_COPLANAR7=late ;;
        latebnp) export RMHD_PLANAR4_STAR_BATCH=1 RMHD_COPLANAR7=late RMHD_COPLANAR7_P=0 ;;
        latebnj) export RMHD_PLANAR4_STAR_BATCH=1 RMHD_COPLANAR7=late RMHD_PLANAR_NJIT=1 ;;
        latebck) export RMHD_PLANAR4_STAR_BATCH=1 RMHD_COPLANAR7=late RMHD_PLANAR_NJIT=check ;;
        latebnjm) export RMHD_PLANAR4_STAR_BATCH=1 RMHD_COPLANAR7=late RMHD_PLANAR_NJIT=1 RMHD_MERGE_SWEEPS=1 ;;
        *) echo "unknown ARM $ARM"; exit 2 ;;
    esac
    export RMHD_ML_CKPT=$RMHD/data/ml_guess_rotor_ft.pt
    export RMHD_ML_CKPTS=$RMHD/data/ml_guess_rotor_big_s42.pt,$RMHD/data/ml_guess_rotor_big_s47.pt
    . "$RMHD/scripts/kernels.env"
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export NUMBA_NUM_THREADS=2 TORCH_THREADS=8 RMHD_POOL=32 RMHD_POOL_THREADS=2 RMHD_POOL_CHUNKS=32
    export PROBLEM RUNG_TIMES=$OUT/rung_times.json RUNG_STATS=$OUT/rung_stats.npz
    rm -rf "$OUT"
    mkdir -p "$OUT"
    cp "$S/$RST" "$OUT/restart.npz"
    echo "== $(date) $(hostname) $PROBLEM ARM=$ARM: HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $RMHD rev-parse --short HEAD)+patch"
    env | grep -E "^RMHD_" | sort | tr "\n" " "; echo
    "$PY" -u "$S/run_2d_lever.py" --problem "$PROBLEM" --n 64 --solver exact --tend "$TEND" \
        --nsnap 8 --tau-weak 1e-2 --tau-bt 1e-9 --exact-retries 2 \
        --exact-max-iter 40 --out "$OUT" --harvest "$OUT/harvest" \
        --restart-every 10 --log-every 1 --restart "$OUT/restart.npz" --max-steps "$STEPS" --harvest-all \
        > "$OUT/run.log" 2>&1
    echo "== $(date) exit $?"
    grep -E "^\s+step" "$OUT/run.log" | tail -n 2
    cat "$RUNG_TIMES"
done
