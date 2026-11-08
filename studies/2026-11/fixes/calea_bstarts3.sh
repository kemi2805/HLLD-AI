#!/bin/bash
#SBATCH --job-name=bstarts3
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Third pass on the 249 faces both fixes still lose: planar4 with the signed-field slow wave (kind S) and the old steerings, central differences, from HLLD p*.
# three start strengths) on the 4Q 128^2 ledger faces the loose-retry fix
# still loses, plus 600 controls; 64 shards, one core each.  rmhd from the
# worktree rmhd_final-loose (loose retry ON; planar4 takes a start).
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz
R=$BASE/HLLD/results/fixes/replay_fix3
OUT=$BASE/HLLD/results/fixes/bstarts3
RMHD=$BASE/rmhd_final-loose2
mkdir -p "$OUT"
export HLLD_DIR=$BASE/HLLD PYTHONPATH=$RMHD RMHD_SLOWSHOCK_LOOSE_RETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RMHD_POOL=0
. "$RMHD/scripts/kernels.env"
cd "$BASE/HLLD" || exit 1
"$PY" -c 'import rmhd, sys; sys.exit(0 if "rmhd_final-loose2" in rmhd.__file__ else 1)' || { echo "PREFLIGHT FAILED: rmhd"; exit 2; }
grep -q KIND_S "$RMHD/rmhd/batched/planar4_b.py" || { echo "PREFLIGHT FAILED: planar4 patch"; exit 2; }
echo "== $(date) $(hostname)"
pids=()
for ((i = 0; i < 64; i++)); do
    "$PY" -u "$S/b_starts.py" "$POP" "$R/replay_off.npz" "$R/replay_on.npz" "$OUT/shard_$(printf %03d $i).npz" \
        --shard $i --nshards 64 --controls 600 --max-iter 40 --kinds SS,SP,PS,SB,BS,PP,PB,BP --deltas 1e-4,1e-3 --central > "$OUT/shard_$(printf %03d $i).log" 2>&1 &
    pids+=($!)
done
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) done, $nf shards failed"
