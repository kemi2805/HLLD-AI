#!/bin/bash
#SBATCH --job-name=rotor-hlld
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The 2D rotor with an APPROXIMATE flux (HLLD by default) on one calea node.
#
# Separate from calea_rotor_exact.sh on purpose: that script's preflight
# imports the rmhd solver package, validates ML checkpoints and sizes a
# worker pool, none of which a run without the exact solver needs -- and all
# of which can fail and cost a node allocation for nothing.
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD
#   N=128 LIMITER=mp5 sbatch scripts/calea_rotor_hlld.sh
#   N=512 LIMITER=mp5 MAX_STEPS=3 sbatch --time=0:30:00 scripts/calea_rotor_hlld.sh
#
# Knobs (env): N (128), LIMITER (mc), SOLVER (hlld; hlle/hllc also work),
# TEND (0.4), NSNAP (8), CFL (0.25), OUT (derived, and run_2d appends the
# limiter for anything but mc), TORCH_THREADS (8), RESUME (1: continue from
# $OUT/restart.npz when present), MAX_STEPS, RESTART_EVERY (10), LOG_EVERY
# (10), PY, HLLD_DIR.  DRYRUN=1 runs the preflight only.
#
# The reconstruction sets the ghost count (PLM 2, MP5/WENO5-Z 3, MP7 4), so
# the preflight asks the code for it: --limiter is a free string in
# run_2d.py, and a typo would otherwise die only after the node is allocated.
set -u
N=${N:-128}
LIMITER=${LIMITER:-mc}
SOLVER=${SOLVER:-hlld}
TEND=${TEND:-0.4}
NSNAP=${NSNAP:-8}
CFL=${CFL:-0.25}
RESUME=${RESUME:-1}
MAX_STEPS=${MAX_STEPS:-}
RESTART_EVERY=${RESTART_EVERY:-10}
LOG_EVERY=${LOG_EVERY:-10}

BASE=/mnt/rafast/miler/codes/rotor2d
PY=${PY:-$HOME/venv/rmhd/bin/python}
cd "${HLLD_DIR:-$BASE/HLLD}" || exit 1
TMPDIR_ERR=$(mktemp)
trap 'rm -f "$TMPDIR_ERR"' EXIT
# (this needs the nodes' Python: the login node has no torch, so DRYRUN on
# kappa reports the import error rather than a verdict)
NG=$("$PY" -c "from src.physics.reconstruction import ghosts_needed; print(ghosts_needed('$LIMITER'))" 2>$TMPDIR_ERR) || {
    echo "PREFLIGHT FAILED for limiter '$LIMITER':"; sed 's/^/    /' $TMPDIR_ERR
    echo "    (known limiters: pcm, mc, minmod, weno5z, mp5, mp7)"; exit 2; }
OUT=${OUT:-results/rotor_${N}_${SOLVER}$([ "$LIMITER" = mc ] || echo "_$LIMITER")}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export TORCH_THREADS=${TORCH_THREADS:-8}
[ "${DRYRUN:-0}" = 1 ] && {
    echo "dry run ok: N=$N solver=$SOLVER limiter=$LIMITER (ng $NG) -> $OUT"; exit 0; }

mkdir -p logs "$OUT"
LOG=${LOG:-logs/rotor_${N}_${SOLVER}_${LIMITER}_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "$LOG") 2>&1

RESTART=""
if [ "$RESUME" = 1 ] && [ -f "$OUT/restart.npz" ]; then
    RESTART="--restart $OUT/restart.npz"
fi

echo "== $(date)  host=$(hostname)  N=$N  solver=$SOLVER  limiter=$LIMITER (ng $NG)  tend=$TEND  out=$OUT  ${RESTART:-fresh start}"
echo "== HLLD $(git rev-parse --short HEAD)"
$PY -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"

$PY -u scripts/run_2d.py --problem rotor --n "$N" --solver "$SOLVER" \
    --limiter "$LIMITER" --tend "$TEND" --nsnap "$NSNAP" --cfl "$CFL" \
    --out "$OUT" --restart-every "$RESTART_EVERY" --log-every "$LOG_EVERY" \
    $RESTART $( [ -n "$MAX_STEPS" ] && echo --max-steps "$MAX_STEPS" )
echo "== $(date)  exit $?"
