#!/bin/bash
#SBATCH --job-name=rotor-exact
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/rotor_exact_%j.out
#SBATCH --error=logs/rotor_exact_%j.err
# ==========================================================================
#  The magnetic rotor with the EXACT Riemann solver as the flux function
#  (plan 4.3 / 4.4).  One node's CORES: the solver runs through the four
#  compiled kernels (numba prange); the GPUs are of no use to it (measured:
#  an MI210 is 5x slower than one core at every batch size).  Partition
#  `gpu` (64 cores, whole MI210 node) starts immediately; `general1` (80
#  cores) was estimated 9 days out on 2026-09-06, `general2`/`fuchs` (40
#  cores) 4-7 h.  Pilot on `gpu_test` (8 h, --gres=gpu:1) or `test` (2 h).
#
#  Cost basis (Mac, 2026-09-06, all four kernels): 28 s per RK3 step at
#  32^2 = 4.7 s per sweep with ~55 attempted interfaces.  The attempted
#  count scales with the front length (~2x per refinement) and the sweep
#  count with 1/dx: 1452 sweeps at 64^2, 2904 at 128^2.  Expect hours at
#  64^2 and about a day at 128^2 on 80 cores; the 48 h limit is margin.
#
#  Usage:   S=/work/astro/miler/codes/rmhd_final/scripts/sbatch_checked.sh
#           $S scripts/goethe_rotor_exact.sbatch                  # 64^2
#           N=128 $S scripts/goethe_rotor_exact.sbatch            # 128^2
#           N=32 TEND=0.02 NUMBA_THREADS=8 $S scripts/goethe_rotor_exact.sbatch -p test -t 1:00:00   # pilot
#  Knobs:   RETRIES (default 2), TEND (0.4), NSNAP (8), OUT, HARVEST_ALL (1),
#           PY, HLLD_DIR, MAX_STEPS (a pilot), POOL/POOL_THREADS, RMHD_ML_CKPT
#           (primary seed), RMHD_ML_CKPTS (comma-separated extra checkpoints
#           whose predictions seed retries 1, 2, ... -- the ensemble
#           multistart; measured paired on identical inputs 2026-09-06:
#           2 ensemble retries 141/295 vs 2 jitter retries 135 vs none 130).
#
#  Judge the result with scripts/rotor_compare.py against results/rotor_64_hlld
#  and results/rotor_128_hlld (the claim under test: exact at N matches HLLD
#  at 2N; the HLLD 64->128 L1(rho) gap is 32.8%).
# ==========================================================================
set -u
N=${N:-64}
TEND=${TEND:-0.4}
NSNAP=${NSNAP:-8}
RETRIES=${RETRIES:-2}
OUT=${OUT:-results/rotor_${N}_exact}

PY=${PY:-/work/astro/miler/python-env/rmhd-cpu/bin/python}
cd "${HLLD_DIR:-/work/astro/miler/codes/HLLD}" || exit 1
# The solver is the installed `rmhd` package.  Asking the interpreter where it
# lives is both the preflight and the only path this script needs.
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd -- uv pip install -e <rmhd_final>"; exit 2; }
export RMHD_ML_CKPT=${RMHD_ML_CKPT:-data/ml_guess_rotor_ft.pt}
# the run-1 snapshots (epoch 150 of job 1767483): fixed files, unlike the
# live checkpoints the 72 h continuation keeps overwriting
export RMHD_ML_CKPTS=${RMHD_ML_CKPTS-data/ml_guess_rotor_big_run1_s42.pt,data/ml_guess_rotor_big_run1_s47.pt}
for c in $RMHD_ML_CKPT ${RMHD_ML_CKPTS//,/ }; do       # relative = inside $RMHD, as exact_flux resolves them
    case "$c" in /*) f=$c ;; *) f=$RMHD/$c ;; esac
    [ -f "$f" ] || { echo "PREFLIGHT FAILED: no checkpoint $f"; exit 2; }
done
HARVEST_ALL=${HARVEST_ALL:-1}     # 1: every solved lane + the per-sweep coverage map
# the four compiled kernels: fan integrators, Alfven pinned solve, slow shock,
# fast/slow shock chain -- each measured against its numpy twin (bitwise for
# the last two) and on the rotor step-1 configuration
. "$RMHD/scripts/kernels.env"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=${NUMBA_THREADS:-16}
export TORCH_THREADS=${TORCH_THREADS:-8}
# the worker pool (src/physics/exact_pool.py): POOL processes, each with
# POOL_THREADS numba threads, sharing the sweep's lanes.  Off (in-process)
# unless POOL >= 2.  On a 40-core node: POOL=20 POOL_THREADS=2.
export RMHD_POOL=${POOL:-0} RMHD_POOL_THREADS=${POOL_THREADS:-2}
export RMHD_POOL_CHUNKS=${POOL_CHUNKS:-${POOL:-0}}
RESUME=${RESUME:-1}     # continue from $OUT/restart.npz when it exists
MAX_STEPS=${MAX_STEPS:-} # a pilot: stop after this many steps (restart file written)

# Submitted through scripts/sbatch_checked.sh this runs on the login node and
# stops here, so a bad path costs a second instead of the queue wait.
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: rmhd at $RMHD, N=$N, pool=${RMHD_POOL}x${RMHD_POOL_THREADS}"; exit 0; }
mkdir -p logs "$OUT"

echo "== $(date)  host=$(hostname)  N=$N  tend=$TEND  retries=$RETRIES  out=$OUT"
echo "== HLLD $(git rev-parse --short HEAD)   rmhd_final $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== ckpt=$RMHD_ML_CKPT  extra=$RMHD_ML_CKPTS  kernels: FAN=$RMHD_FAN ALFVEN=$RMHD_ALFVEN SLOWSHOCK=$RMHD_SLOWSHOCK SHOCK=$RMHD_SHOCK  numba threads=$NUMBA_NUM_THREADS  pool=${RMHD_POOL}x${RMHD_POOL_THREADS}"
RESTART=""
if [ "$RESUME" = 1 ] && [ -f "$OUT/restart.npz" ]; then
    RESTART="--restart $OUT/restart.npz"; echo "== resuming from $OUT/restart.npz"
fi
$PY -c "import numpy, numba, torch; print('numpy', numpy.__version__, 'numba', numba.__version__, 'torch', torch.__version__)"

$PY -u scripts/run_2d.py --problem rotor --n "$N" --solver exact --tend "$TEND" \
    --nsnap "$NSNAP" --tau-weak 1e-2 --tau-bt 1e-9 --exact-retries "$RETRIES" \
    --exact-max-iter 40 --out "$OUT" --harvest "$OUT/harvest" \
    --restart-every 10 $RESTART \
    $( [ -n "$MAX_STEPS" ] && echo --max-steps "$MAX_STEPS" ) \
    $( [ "$HARVEST_ALL" = 1 ] && echo --harvest-all )
echo "== $(date)  exit $?"
