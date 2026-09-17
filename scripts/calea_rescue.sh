#!/usr/bin/env bash
# Rescue stubborn rotor interfaces across all 64 cores of calea01.
#
# Each worker takes every Nth interface, runs the verified planar-descent
# rescue, and writes its own shard.  Measured yield 5.0% of the stubborn set
# at ~1.3 s per interface, so 630k interfaces is ~3.6 h on 64 workers and
# should produce ~31k verified-exact rows -- the hardest training data the
# project can generate.
set -u
ROOT=${ROOT:-/mnt/rafast/miler/codes/rotor2d/HLLD}
PY=${PY:-$HOME/venv/rmhd/bin/python}
HARVEST=${HARVEST:-$ROOT/results/rotor_64_p5/harvest}
OUT=${OUT:-$ROOT/results/rescued}
MAX=${MAX:-2000000}
N=${N:-64}
cd "$ROOT" || exit 1
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd -- pip install -e <rmhd_final>"; exit 2; }
. "$RMHD/scripts/kernels.env"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -d "$HARVEST" ] || { echo "PREFLIGHT FAILED: no harvest $HARVEST"; exit 2; }
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: rmhd at $RMHD, $N workers on $HARVEST"; exit 0; }
mkdir -p "$OUT" "$OUT/logs"
echo "== $(date)  $N workers on $HARVEST"
for ((i=0; i<N; i++)); do
    nohup "$PY" scripts/rescue_collect.py "$HARVEST" \
        --out "$OUT/rescued_$(printf '%03d' $i).npz" \
        --max "$MAX" --shard "$i" --nshards "$N" \
        > "$OUT/logs/w$(printf '%03d' $i).log" 2>&1 &
done
wait
echo "== $(date)  done"
ls -la "$OUT"/rescued_*.npz 2>/dev/null | wc -l
