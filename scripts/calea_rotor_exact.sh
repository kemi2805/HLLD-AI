#!/bin/bash
# The 2D rotor with the exact RMHD Riemann solver on calea01 (ITP): one
# 64-core Ice Lake node, no scheduler.  Same run as goethe_rotor_exact.sbatch,
# with the solver's per-lane pipeline spread over RMHD_POOL worker processes
# (src/physics/exact_pool.py) so the node's cores are used -- one process
# uses one core for its Python orchestration, however many numba threads it
# has.
#
#   nohup scripts/calea_rotor_exact.sh > /dev/null 2>&1 &     (log: see LOG)
#
# Knobs (env): N (64), TEND (0.4), NSNAP (8), RETRIES (2), OUT, POOL (32
# workers), POOL_THREADS (2 numba threads each), POOL_CHUNKS (= POOL),
# TORCH_THREADS (8), RESUME (1: continue from $OUT/restart.npz when present),
# MAX_STEPS (a pilot: stop after this many steps), RMHD_ML_CKPT/RMHD_ML_CKPTS,
# PY, HLLD_DIR.  DRYRUN=1 runs the preflight only.
set -u
N=${N:-64}
TEND=${TEND:-0.4}
NSNAP=${NSNAP:-8}
RETRIES=${RETRIES:-2}
OUT=${OUT:-results/rotor_${N}_exact}
POOL=${POOL:-32}
POOL_THREADS=${POOL_THREADS:-2}
POOL_CHUNKS=${POOL_CHUNKS:-$POOL}
RESUME=${RESUME:-1}
MAX_STEPS=${MAX_STEPS:-}
RESTART_EVERY=${RESTART_EVERY:-10}
LOG_EVERY=${LOG_EVERY:-1}

BASE=/mnt/rafast/miler/codes/rotor2d
PY=${PY:-$HOME/venv/rmhd/bin/python}
cd "${HLLD_DIR:-$BASE/HLLD}" || exit 1
# The solver is the installed `rmhd` package.  Asking the interpreter where it
# lives is both the preflight and the only path this script needs.
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd -- pip install -e $BASE/rmhd_final"; exit 2; }
export RMHD_ML_CKPT=${RMHD_ML_CKPT:-data/ml_guess_rotor_ft.pt}
export RMHD_ML_CKPTS=${RMHD_ML_CKPTS-data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt}
for c in $RMHD_ML_CKPT ${RMHD_ML_CKPTS//,/ }; do       # relative = inside $RMHD, as exact_flux resolves them
    case "$c" in /*) f=$c ;; *) f=$RMHD/$c ;; esac
    [ -f "$f" ] || { echo "PREFLIGHT FAILED: no checkpoint $f"; exit 2; }
done
HARVEST_ALL=${HARVEST_ALL:-1}
. "$RMHD/scripts/kernels.env"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=${NUMBA_THREADS:-$POOL_THREADS}   # the main process; workers set their own
export TORCH_THREADS=${TORCH_THREADS:-8}
export RMHD_POOL=$POOL RMHD_POOL_THREADS=$POOL_THREADS RMHD_POOL_CHUNKS=$POOL_CHUNKS
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: rmhd at $RMHD, N=$N, pool=${POOL}x${POOL_THREADS}"; exit 0; }

mkdir -p logs "$OUT"
LOG=${LOG:-logs/rotor_${N}_exact_$(date +%Y%m%d_%H%M%S).log}
exec > >(tee -a "$LOG") 2>&1

RESTART=""
if [ "$RESUME" = 1 ] && [ -f "$OUT/restart.npz" ]; then
    RESTART="--restart $OUT/restart.npz"
fi

echo "== $(date)  host=$(hostname)  N=$N  tend=$TEND  retries=$RETRIES  out=$OUT  pool=${POOL}x${POOL_THREADS} chunks=$POOL_CHUNKS  ${RESTART:-fresh start}"
echo "== HLLD $(git rev-parse --short HEAD)   rmhd_final $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== ckpt=$RMHD_ML_CKPT  extra=$RMHD_ML_CKPTS  kernels: FAN=$RMHD_FAN ALFVEN=$RMHD_ALFVEN SLOWSHOCK=$RMHD_SLOWSHOCK SHOCK=$RMHD_SHOCK"
$PY -c "import numpy, numba, torch; print('numpy', numpy.__version__, 'numba', numba.__version__, 'torch', torch.__version__)"

$PY -u scripts/run_2d.py --problem rotor --n "$N" --solver exact --tend "$TEND" \
    --nsnap "$NSNAP" --tau-weak 1e-2 --tau-bt 1e-9 --exact-retries "$RETRIES" \
    --exact-max-iter 40 --out "$OUT" --harvest "$OUT/harvest" \
    --restart-every "$RESTART_EVERY" --log-every "$LOG_EVERY" $RESTART \
    $( [ -n "$MAX_STEPS" ] && echo --max-steps "$MAX_STEPS" ) \
    $( [ "$HARVEST_ALL" = 1 ] && echo --harvest-all )
echo "== $(date)  exit $?"
