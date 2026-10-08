#!/bin/bash
#SBATCH --job-name=ledger
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The failure ledger (science plan H, step 1) on one calea node:
#   1. record   sweeps of the developed rotor, production settings
#   2. collect  replay them through production, take the failing lanes
#   3. run      offer every failing lane to every method we have
#   4. the ledger
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD
#   sbatch scripts/calea_failure_ledger.sh
#   CHECK=1 sbatch --time=1:00:00 -c 8 --exclusive=no scripts/calea_failure_ledger.sh
#
# Knobs (env): HARVEST (an exact run's harvest dir: skip the recording and
# take that run's own lost/solved faces, NFAIL/NCTRL of each), PROBLEM (rotor; orszag_tang, riemann2d -- the recorder's
# problem, gamma 5/3 for the ledger's own stages, so not orszag_tang unless
# those are made gamma-aware), B0 (riemann2d's field), N (64), NSTEP (2), WINDOWS ("pfa:0.10 pfb:0.25 pfc:0.40"; the
# held-out set is recorded by a second submission with
# WINDOWS="pfd:0.175 pfe:0.325" OUT=...), OUT, NW (workers, 64), HLLD_DIR, PY.
# CHECK=1: a 32^2, one-step, few-lane pass through every stage -- the path
# check before the node is spent (CHECK_LIMIT=0: every lane of it).
#
# Output: $OUT/ledger.txt (the totals), $OUT/cards.txt (one card per failing
# interface: states, physics, what production did, every attempt of every
# method, what the tube saw, the verdict), $OUT/cards.csv (one line each),
# and $OUT/logs/*.log, which carry a line per interface as it is processed.
#
# Fixed IN ADVANCE, and not to be extended after the numbers are seen: three
# windows x two steps x six sweeps = 36 sweeps.
set -u
PROBLEM=${PROBLEM:-rotor}
N=${N:-64}
NSTEP=${NSTEP:-2}
WINDOWS=${WINDOWS:-"pfa:0.10 pfb:0.25 pfc:0.40"}
NW=${NW:-64}
LIMIT=""
BASE=/mnt/rafast/miler/codes/rotor2d
PY=${PY:-$HOME/venv/rmhd/bin/python}
cd "${HLLD_DIR:-$BASE/HLLD}" || exit 1
if [ "${CHECK:-0}" = 1 ]; then
    N=32; NSTEP=1; WINDOWS="chk:0.05"; NW=${NW_CHECK:-4}
    # CHECK_LIMIT=0 offers every lane: a limit checks the PATH, never the
    # numbers (the first check's "0 recovered" was its 6-lane limit)
    [ "${CHECK_LIMIT:-6}" = 0 ] || LIMIT="--limit ${CHECK_LIMIT:-6}"
    OUT=${OUT:-/mnt/rafast/miler/scratch/b/ledger_check}
    rm -rf "$OUT"
fi
OUT=${OUT:-results/failure_ledger_$N}
[ -e "$OUT/population.npz" ] && { echo "PREFLIGHT FAILED: $OUT already holds a ledger"; exit 2; }
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd"; exit 2; }
"$PY" -c "from rmhd.batched import planar5_b as P; P.default_seed, P.unk6_to_unk3" ||
    { echo "PREFLIGHT FAILED: the installed rmhd predates the planar seed surface"; exit 2; }
export RMHD_ML_CKPT=${RMHD_ML_CKPT:-data/ml_guess_rotor_ft.pt}
export RMHD_ML_CKPTS=${RMHD_ML_CKPTS-data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt}
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/rec" "$OUT/logs" logs
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- src scripts)" ] || echo +dirty)  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== problem=$PROBLEM N=$N NSTEP=$NSTEP windows: $WINDOWS -> $OUT  workers=$NW"

# ── 1. record, the windows side by side ──────────────────────────────────
. "$RMHD/scripts/kernels.env"
if [ -z "${HARVEST:-}" ]; then
nwin=$(echo $WINDOWS | wc -w)
pool=$(( NW / nwin > 2 ? NW / nwin : 0 ))
pids=()
for w in $WINDOWS; do
    tag=${w%%:*}; t0=${w##*:}
    ( RMHD_POOL=$pool RMHD_POOL_THREADS=1 NUMBA_NUM_THREADS=2 TORCH_THREADS=4 \
      PROBLEM=$PROBLEM B0=${B0:-0.5} N=$N T0=$t0 NSTEP=$NSTEP REC_DIR=$OUT/rec \
      "$PY" -u scripts/rotor_replay.py record $tag > "$OUT/logs/record_$tag.log" 2>&1 ) &
    pids+=($!)
done
nfail=0; for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
for w in $WINDOWS; do tail -n 1 "$OUT/logs/record_${w%%:*}.log"; done
[ "$nfail" = 0 ] || { echo "RECORD FAILED ($nfail)"; exit 3; }

fi

# ── 2. collect: production on the recordings, and G0 ─────────────────────
# (HARVEST=<run>/harvest: the run's own lost and solved faces instead of
#  recorded windows -- NFAIL / NCTRL of each, replayed; see collect_harvest)
if [ -n "${HARVEST:-}" ]; then
  RMHD_POOL=$(( NW / 2 )) RMHD_POOL_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_THREADS=8 \
      "$PY" -u scripts/failure_ledger.py collect-harvest "$HARVEST" \
      --n-fail "${NFAIL:-1200}" --n-ctrl "${NCTRL:-1200}" \
      --out "$OUT/population.npz" 2>&1 | tee "$OUT/logs/collect.log" | tail -n 6
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "COLLECT-HARVEST FAILED"; exit 4; }
else
tags=$(for w in $WINDOWS; do echo -n "${w%%:*} "; done)
RMHD_POOL=$(( NW / 2 )) RMHD_POOL_THREADS=2 NUMBA_NUM_THREADS=2 TORCH_THREADS=8 \
    "$PY" -u scripts/failure_ledger.py collect "$OUT/rec" $tags \
    --out "$OUT/population.npz" 2>&1 | tee "$OUT/logs/collect.log" | tail -n 6
[ "${PIPESTATUS[0]}" = 0 ] || { echo "COLLECT FAILED (G0?)"; exit 4; }
fi

# ── 3. run: the elementary methods and the tube label, compiled kernels ──
export NUMBA_NUM_THREADS=1 RMHD_POOL=0
stage() {   # stage <name> <methods>
    local pids=() nfail=0
    for ((i = 0; i < NW; i++)); do
        "$PY" -u scripts/failure_ledger.py run "$OUT/population.npz" \
            --methods "$2" --shard "$i" --nshards "$NW" $LIMIT \
            --out "$OUT/shard_$1_$(printf '%03d' $i).npz" \
            > "$OUT/logs/$1_$(printf '%03d' $i).log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
    echo "== $(date) stage $1 ($2): $(ls "$OUT"/shard_$1_*.npz 2>/dev/null | wc -l)/$NW shards, $nfail failed"
    return $nfail
}
stage el a,b,c,e || exit 5
# the intermediate branch: the window patch reaches the numpy kernel only
unset RMHD_SLOWSHOCK
stage im d || exit 6

# ── 4. the ledger ─────────────────────────────────────────────────────────
"$PY" scripts/failure_ledger.py --summarise "$OUT" | tee "$OUT/ledger.txt"
# one card per failing interface, and the one-line-per-interface table
"$PY" scripts/failure_ledger.py --report "$OUT" --group failing \
    --out "$OUT/cards.txt"
echo LEDGER DONE
