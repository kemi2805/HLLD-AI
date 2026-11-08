#!/bin/bash
#SBATCH --job-name=replay-final2
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired replay of the 4Q 128^2 failure-ledger population (78407), all loss
# fixes of 2026-11-04 (loose retry, planar4 from HLLD p* with forward then
# central differences, the signed-field slow wave), three arms side by side:
#   off   both switches 0 (production)
#   lean9 on, the lean start set (5 starts, 25 iterations), entropy may fall by 1e-9
#   lean4 the same with 1e-4
# rmhd rmhd_final-loose2, HLLD HLLD-fix2 (patches fixes/kind_s_*.diff).
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=${POP:-$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz}
OUT=${OUT:-$BASE/HLLD/results/fixes/replay_final2}
RMHD=$BASE/rmhd_final-loose5
HD=$BASE/HLLD-fix5
mkdir -p "$OUT"
export HLLD_DIR=$HD PYTHONPATH=$RMHD
cd "$HD" || exit 1
"$PY" -c 'import rmhd, sys; sys.exit(0 if "rmhd_final-loose5" in rmhd.__file__ else 1)' ||
    { echo "PREFLIGHT FAILED: rmhd does not come from the worktree"; exit 2; }
grep -q KIND_S "$RMHD/rmhd/batched/planar4_b.py" || { echo "PREFLIGHT FAILED: rmhd patch"; exit 2; }
grep -q _PLANAR4_STAR_SIDE src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
export RMHD_ML_CKPT=$BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=$BASE/rmhd_final/data/ml_guess_rotor_big_s42.pt,$BASE/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
. "$RMHD/scripts/kernels.env"
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)+patch  rmhd $(git -C $RMHD rev-parse --short HEAD)+patch  pop $POP"
run() {   # run <arm> <switch> <ds>
    RMHD_SLOWSHOCK_LOOSE_RETRY=$2 RMHD_PLANAR4_STAR=$2 RMHD_PLANAR4_STAR_DS=$3 \
        RMHD_POOL=10 RMHD_POOL_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_THREADS=2 \
        "$PY" -u "$S/replay_pop.py" "$POP" "$OUT/replay_$1.npz" --chunk 512 \
        > "$OUT/replay_$1.log" 2>&1 &
}
run off 0 1e-9
run final9 1 1e-9
run final4 1 1e-4
wait
for a in off final9 final4; do grep -v -i warn "$OUT/replay_$a.log" | tail -n 1; done
echo "== $(date) done"
