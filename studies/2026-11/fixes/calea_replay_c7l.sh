#!/bin/bash
#SBATCH --job-name=replay-c7l
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The planar seven-wave stage (RMHD_COPLANAR7) where it runs: first / after / late (a rung after the five-wave solver), on the 4Q 128^2 failure-ledger
# population (78407), four arms side by side (user 2026-11-05: "take a look on the
# entropy fall ..."; 11-05: "give the seven-wave solver signed-field slow waves and the rotation. be ambitious"):
#   finalrot     the loss fixes + the gate + the rotation re-solve
#   finalrotc7   the same + the planar seven-wave stage first (RMHD_COPLANAR7=1)
#   finalrotc7a  ... after the six-unknown Newton (=after)
#   finalrotc7l  ... as a rung after the five-wave solver (=late); every arm with the batched starts
# rmhd rmhd_final-loose9, HLLD HLLD-fix10 (c7_patches, setup_late.sh).
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz
OUT=$BASE/HLLD/results/fixes/replay_c7l
RMHD=$BASE/rmhd_final-loose9
HD=$BASE/HLLD-fix10
mkdir -p "$OUT"
export HLLD_DIR=$HD PYTHONPATH=$RMHD
cd "$HD" || exit 1
grep -q _COPLANAR7_LATE src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
export RMHD_ML_CKPT=$BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=$BASE/rmhd_final/data/ml_guess_rotor_big_s42.pt,$BASE/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
. "$RMHD/scripts/kernels.env"
echo "== $(date) $(hostname)"
run() {   # run <arm> <fixes 0|1> <rotation 0|1> <coplanar7 0|1|after|late> <pressure stage 0|1>   (the gate and the batched starts always on)
    RMHD_SLOWSHOCK_LOOSE_RETRY=$2 RMHD_PLANAR4_STAR=$2 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1 RMHD_PLANAR4_ROT=$3 RMHD_COPLANAR7=$4 RMHD_COPLANAR7_P=$5 RMHD_PLANAR4_STAR_BATCH=1 \
        RMHD_POOL=7 RMHD_POOL_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_THREADS=2 \
        "$PY" -u "$S/replay_pop.py" "$POP" "$OUT/replay_$1.npz" --chunk 512 \
        > "$OUT/replay_$1.log" 2>&1 &
}
run finalrot 1 1 0 1
run finalrotc7 1 1 1 1
run finalrotc7a 1 1 after 1
run finalrotc7l 1 1 late 1
wait
for a in finalrot finalrotc7 finalrotc7a finalrotc7l; do echo "$a: $(grep -v -i warn "$OUT/replay_$a.log" | tail -n 1)"; done
echo "== $(date) done"
