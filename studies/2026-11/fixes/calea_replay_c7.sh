#!/bin/bash
#SBATCH --job-name=replay-c7
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The planar seven-wave stage (RMHD_COPLANAR7) on the 4Q 128^2 failure-ledger
# population (78407), four arms side by side (user 2026-11-05: "take a look on the
# entropy fall ..."; 11-05: "give the seven-wave solver signed-field slow waves and the rotation. be ambitious"):
#   finalrot     the loss fixes + the gate + the rotation re-solve (the current candidate)
#   finalrotc7   the same + the planar seven-wave stage
#   finalrotc7np the same without its pressure-steered stage (RMHD_COPLANAR7_P=0)
#   prodgatec7   production + the gate + the planar seven-wave stage (no other fix)
# rmhd rmhd_final-loose7, HLLD HLLD-fix8 (c7_patches).
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz
OUT=$BASE/HLLD/results/fixes/replay_c7
RMHD=$BASE/rmhd_final-loose7
HD=$BASE/HLLD-fix8
mkdir -p "$OUT"
export HLLD_DIR=$HD PYTHONPATH=$RMHD
cd "$HD" || exit 1
grep -q coplanar7 src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
export RMHD_ML_CKPT=$BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=$BASE/rmhd_final/data/ml_guess_rotor_big_s42.pt,$BASE/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
. "$RMHD/scripts/kernels.env"
echo "== $(date) $(hostname)"
run() {   # run <arm> <fixes 0|1> <rotation 0|1> <coplanar7 0|1> <pressure stage 0|1>   (the gate always on)
    RMHD_SLOWSHOCK_LOOSE_RETRY=$2 RMHD_PLANAR4_STAR=$2 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1 RMHD_PLANAR4_ROT=$3 RMHD_COPLANAR7=$4 RMHD_COPLANAR7_P=$5 \
        RMHD_POOL=7 RMHD_POOL_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_THREADS=2 \
        "$PY" -u "$S/replay_pop.py" "$POP" "$OUT/replay_$1.npz" --chunk 512 \
        > "$OUT/replay_$1.log" 2>&1 &
}
run finalrot 1 1 0 1
run finalrotc7 1 1 1 1
run finalrotc7np 1 1 1 0
run prodgatec7 0 0 1 1
wait
for a in finalrot finalrotc7 finalrotc7np prodgatec7; do echo "$a: $(grep -v -i warn "$OUT/replay_$a.log" | tail -n 1)"; done
echo "== $(date) done"
