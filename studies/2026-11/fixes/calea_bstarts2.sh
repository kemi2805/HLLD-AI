#!/bin/bash
#SBATCH --job-name=bstarts2
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Second pass on the 282 faces both fixes still lose: 200 iterations, starts PP/PB/BP x 3 strengths x p* scaled 1, 0.999, 1.001.
# three start strengths) on the 4Q 128^2 ledger faces the loose-retry fix
# still loses, plus 600 controls; 64 shards, one core each.  rmhd from the
# worktree rmhd_final-loose (loose retry ON; planar4 takes a start).
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz
R=$BASE/HLLD/results/fixes/replay_fix2
OUT=$BASE/HLLD/results/fixes/bstarts2
RMHD=$BASE/rmhd_final-loose
mkdir -p "$OUT"
export HLLD_DIR=$BASE/HLLD PYTHONPATH=$RMHD RMHD_SLOWSHOCK_LOOSE_RETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RMHD_POOL=0
. "$RMHD/scripts/kernels.env"
cd "$BASE/HLLD" || exit 1
"$PY" -c 'import rmhd, sys; sys.exit(0 if "rmhd_final-loose" in rmhd.__file__ else 1)' || { echo "PREFLIGHT FAILED: rmhd"; exit 2; }
grep -q unk_init "$RMHD/rmhd/batched/planar4_b.py" || { echo "PREFLIGHT FAILED: planar4 patch"; exit 2; }
echo "== $(date) $(hostname)"
pids=()
for ((i = 0; i < 64; i++)); do
    "$PY" -u "$S/b_starts.py" "$POP" "$R/replay_off.npz" "$R/replay_on.npz" "$OUT/shard_$(printf %03d $i).npz" \
        --shard $i --nshards 64 --controls 600 --max-iter 200 --kinds PP,PB,BP --deltas 1e-4,1e-3,1e-2 --pscale 1,0.999,1.001 > "$OUT/shard_$(printf %03d $i).log" 2>&1 &
    pids+=($!)
done
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) done, $nf shards failed"
