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
#  Usage:   sbatch scripts/goethe_rotor_exact.sbatch              # 64^2
#           N=128 sbatch scripts/goethe_rotor_exact.sbatch        # 128^2
#           N=32 TEND=0.02 NUMBA_THREADS=8 sbatch -p test -t 1:00:00 scripts/goethe_rotor_exact.sbatch   # pilot
#  Knobs:   RETRIES (default 2), TEND (0.4), NSNAP (8), OUT, RMHD_ML_CKPT
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

export RMHD_ROOT=${RMHD_ROOT:-/work/astro/miler/codes/rmhd_final}
export RMHD_ML_CKPT=${RMHD_ML_CKPT:-data/ml_guess_rotor_ft.pt}
export RMHD_ML_CKPTS=${RMHD_ML_CKPTS-data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt}
# the four compiled kernels: fan integrators, Alfven pinned solve, slow shock,
# fast/slow shock chain -- each measured against its numpy twin (bitwise for
# the last two) and on the rotor step-1 configuration
export RMHD_FAN=njit RMHD_ALFVEN=njit RMHD_SLOWSHOCK=njit RMHD_SHOCK=njit
# numpy/BLAS single-threaded (its dispatch is the cost, not its arithmetic);
# torch (the CT/HLLD part) a few threads.  numba's thread count is a KNOB
# (NUMBA_THREADS), not the node: the kernels are launched ~50,000 times per
# sweep on batches of ~50 lanes, and with 64 threads the pool spends its time
# spinning at the launches -- the first pilot burned 58 cores for 24 minutes
# without finishing a step (2026-09-06).  Pilot 8 / 16 before trusting more.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=${NUMBA_THREADS:-16}
export TORCH_THREADS=${TORCH_THREADS:-8}

PY=/work/astro/miler/python-env/rmhd-cpu/bin/python
cd /work/astro/miler/codes/HLLD
mkdir -p logs "$OUT"

echo "== $(date)  host=$(hostname)  N=$N  tend=$TEND  retries=$RETRIES  out=$OUT"
echo "== HLLD $(git rev-parse --short HEAD)   rmhd_final $(git -C $RMHD_ROOT rev-parse --short HEAD)"
echo "== ckpt=$RMHD_ML_CKPT  extra=$RMHD_ML_CKPTS  kernels: FAN=$RMHD_FAN ALFVEN=$RMHD_ALFVEN SLOWSHOCK=$RMHD_SLOWSHOCK SHOCK=$RMHD_SHOCK  numba threads=$NUMBA_NUM_THREADS"
$PY -c "import numpy, numba, torch; print('numpy', numpy.__version__, 'numba', numba.__version__, 'torch', torch.__version__)"

$PY -u scripts/run_2d.py --problem rotor --n "$N" --solver exact --tend "$TEND" \
    --nsnap "$NSNAP" --tau-weak 1e-2 --tau-bt 1e-9 --exact-retries "$RETRIES" \
    --exact-max-iter 40 --out "$OUT" --harvest "$OUT/harvest"
echo "== $(date)  exit $?"
