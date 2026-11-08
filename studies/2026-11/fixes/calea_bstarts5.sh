#!/bin/bash
#SBATCH --job-name=bstarts5
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The 41 ledger lanes the FINAL loss-fix configuration still loses (40 near-
# normal strong-field): planar4, central differences, every steering, from
#   side10  a fast wave within 10% of p* started at its ahead pressure x (1 +- 1e-3 / 1e-2)
#   both    both fast waves at their own ahead pressures x (1 +- 1e-3 / 1e-2)
# 3 variants x 64 shards, the controls with each.
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz
R=$BASE/HLLD/results/fixes/replay_final
RMHD=$BASE/rmhd_final-loose4
export HLLD_DIR=$BASE/HLLD PYTHONPATH=$RMHD RMHD_SLOWSHOCK_LOOSE_RETRY=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RMHD_POOL=0
. "$RMHD/scripts/kernels.env"
cd "$BASE/HLLD" || exit 1
"$PY" -c 'import rmhd, sys; sys.exit(0 if "rmhd_final-loose4" in rmhd.__file__ else 1)' || { echo "PREFLIGHT FAILED: rmhd"; exit 2; }
echo "== $(date) $(hostname)"
pids=()
for v in side10 both; do
    OUT=$BASE/HLLD/results/fixes/bstarts5_$v
    mkdir -p "$OUT"
    case $v in
        side10) X="--pmode side --near 0.10" ;;
        both) X="--pmode both" ;;
    esac
    for ((i = 0; i < 32; i++)); do
        "$PY" -u "$S/b_starts.py" "$POP" "$R/replay_off.npz" "$R/replay_final4.npz" "$OUT/shard_$(printf %03d $i).npz" \
            --shard $i --nshards 32 --controls 600 --max-iter 40 --kinds PP,SS,PS,SP,BP,PB --deltas 1e-4 \
            --pscale 1.001,0.999,1.01,0.99 --central $X > "$OUT/shard_$(printf %03d $i).log" 2>&1 &
        pids+=($!)
    done
done
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) done, $nf shards failed"
