#!/bin/bash
#SBATCH --job-name=ledger-r2d128
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The faces the four-quadrant 128^2 exact run with ft3 (77226) LOST -- the anatomy
# study of 2026-11-04 ("why they fail") -- classified
# with the failure ledger (scripts/failure_ledger.py, harvest mode):
#   1. collect-harvest: NFAIL lost and NCTRL solved faces, replayed through
#      today's production path (the replay sets the groups: the four-unknown
#      and degenerate answers, which the harvest does not keep as solved rows,
#      come back exact)
#   2. methods a,b,c (diverse seven-wave seeds; planar, 4 flips x 6 seeds; the
#      eps-ladder and the tube seed) and d (the intermediate branch, numpy
#      slow-shock kernel), then the ledger and the cards: the R1..R4 verdicts
#   (no tube label: method e took 9-17 h per shard on these faces, 10-28)
# HLLD from the worktree $HD = HLLD-harvest, the code the run used (51e9c67 +
# the harvest-cap patch); rmhd is production's (9ca5a4b), as in the run.
#
#   ssh itp; sbatch /mnt/rafast/miler/scratch/r2d_retrain/anatomy/calea_ledger_r2d128.sh
#   CHECK=1: 4 + 4 faces, 2 workers, methods a,b,d -- the path check
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
HD=${HD:-$BASE/HLLD-harvest}
PY=${PY:-$HOME/venv/rmhd/bin/python}
RUN=${RUN:-r2d_128_exact_ml_r2d_it3}
HARVEST=$BASE/HLLD/results/$RUN/harvest
NW=${NW:-64}; NFAIL=${NFAIL:-1500}; NCTRL=${NCTRL:-600}
OUT=${OUT:-$BASE/HLLD/results/failure_ledger_r2d128_ft3}
if [ "${CHECK:-0}" = 1 ]; then
    NW=2; NFAIL=4; NCTRL=4; OUT=/mnt/rafast/miler/scratch/ledger/check_$RUN
    rm -rf "$OUT"
fi
cd "$HD" || exit 1
[ -e "$OUT/population.npz" ] && { echo "PREFLIGHT FAILED: $OUT already holds a ledger"; exit 2; }
[ -d "$HARVEST" ] || { echo "PREFLIGHT FAILED: no $HARVEST"; exit 2; }
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd"; exit 2; }
"$PY" -c 'import sys, rmhd.batched.fullcontact_b as F; sys.exit(0 if F.ALFVEN_EXACT else 1)' ||
    { echo "PREFLIGHT FAILED: the closed-form Alfven rotation is off"; exit 2; }
grep -q "def use_gamma" scripts/failure_ledger.py ||
    { echo "PREFLIGHT FAILED: failure_ledger.py does not take the population's gamma"; exit 2; }
export RMHD_ML_CKPT=$BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt   # the run's primary (ft3)
export RMHD_ML_CKPTS=$RMHD/data/ml_guess_rotor_big_s42.pt,$RMHD/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- src scripts)" ] || echo +dirty), failure_ledger.py md5 $(md5sum scripts/failure_ledger.py | cut -c1-12)  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== $RUN: $NFAIL lost + $NCTRL solved faces -> $OUT, $NW workers"
. "$RMHD/scripts/kernels.env"

# ── 1. collect ────────────────────────────────────────────────────────────
pool=$(( NW / 2 )); [ "$pool" -ge 2 ] || pool=0
RMHD_POOL=$pool RMHD_POOL_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_THREADS=8 \
    "$PY" -u scripts/failure_ledger.py collect-harvest "$HARVEST" \
    --n-fail "$NFAIL" --n-ctrl "$NCTRL" --out "$OUT/population.npz" \
    > "$OUT/logs/collect.log" 2>&1
rc=$?
grep -v -i warn "$OUT/logs/collect.log" | tail -n 6
[ "$rc" = 0 ] || { echo "COLLECT-HARVEST FAILED"; exit 4; }

# ── 2, 3. the methods, one shard per worker ───────────────────────────────
export NUMBA_NUM_THREADS=1 RMHD_POOL=0
stage() {   # stage <name> <methods>
    local pids=() nfail=0 i
    for ((i = 0; i < NW; i++)); do
        "$PY" -u scripts/failure_ledger.py run "$OUT/population.npz" \
            --methods "$2" --shard "$i" --nshards "$NW" \
            --out "$OUT/shard_$1_$(printf '%03d' $i).npz" \
            > "$OUT/logs/$1_$(printf '%03d' $i).log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
    echo "== $(date) stage $1 ($2): $(ls "$OUT"/shard_$1_*.npz 2>/dev/null | wc -l)/$NW shards, $nfail failed"
}
ledger() {
    "$PY" scripts/failure_ledger.py --summarise "$OUT" 2>&1 | grep -v -i warn > "$OUT/ledger.txt"
    cat "$OUT/ledger.txt"
    "$PY" scripts/failure_ledger.py --report "$OUT" --group all --out "$OUT/cards.txt" 2>&1 |
        grep -v -i warn | tail -n 1
}
if [ "${CHECK:-0}" = 1 ]; then
    stage el a,b
    ( unset RMHD_SLOWSHOCK; stage im d )
    ledger
    echo "== $(date) CHECK done"
    exit 0
fi
stage el a,b,c
( unset RMHD_SLOWSHOCK; stage im d )      # the window patch reaches the numpy kernel only
# no tube stage (e): on the four-quadrant faces it took 9-17 h per shard (10-28)
echo "== $(date) the ledger"
ledger
echo "== $(date) done"
