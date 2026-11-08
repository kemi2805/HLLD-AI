#!/bin/bash
#SBATCH --job-name=compound-r2d128
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Can a compound slow wave answer the faces the four-quadrant 128^2 run lost?
# (user 2026-11-04: "could we solve them with the compound wave?")
# compound_test.py on the failure ledger 78407's 872 failing lanes and 192 of
# its controls, 64 shards side by side.  HLLD scripts/compound_wave.py's
# element (numpy slow-shock path: NO kernels.env, RMHD_SLOWSHOCK unset).
set -u
S=/mnt/rafast/miler/scratch/r2d_retrain/anatomy
LED=/mnt/rafast/miler/codes/rotor2d/HLLD/results/failure_ledger_r2d128_ft3
OUT=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/anatomy128/compound
PY=$HOME/venv/rmhd/bin/python
NW=${NW:-64}
export HLLD_DIR=/mnt/rafast/miler/codes/rotor2d/HLLD-harvest
unset RMHD_SLOWSHOCK RMHD_FAN RMHD_ALFVEN RMHD_SHOCK
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "$OUT/logs"
echo "== $(date) $(hostname)  HLLD-harvest $(git -C $HLLD_DIR rev-parse --short HEAD)"
pids=()
for ((i = 0; i < NW; i++)); do
    "$PY" -u "$S/compound_test.py" "$LED" "$OUT/shard_$(printf '%03d' $i).npz" --shard $i --nshards $NW \
        > "$OUT/logs/shard_$(printf '%03d' $i).log" 2>&1 &
    pids+=($!)
done
nfail=0; for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
echo "== $(date) $(ls $OUT/shard_*.npz | wc -l)/$NW shards, $nfail failed"
grep -h "done:" $OUT/logs/shard_*.log | head -n 70
