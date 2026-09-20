#!/usr/bin/env bash
#SBATCH --job-name=tubefeat
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The high-resolution tube test (scripts/tube_features.py) on one calea node.
# Submit from iota:
#   cd /mnt/rafast/miler/codes/rotor2d/HLLD
#   N=100 TAG=pilot sbatch scripts/calea_tube_features.sh
#
# Knobs: N (lanes per group, 0 = all), NW (workers, default the granted
# cores), TAG, LANES (a coplanar_limit output directory -- that run's classes
# are what the groups are drawn from), ROOT, EXTRA (e.g. "--ncells 512").
# The tubes are torch on one thread each, so the node's cores are the batch.
set -u
ROOT=${ROOT:-/mnt/rafast/miler/codes/rotor2d/HLLD}
PY=${PY:-$HOME/venv/rmhd/bin/python}
N=${N:-100}
NW=${NW:-$(nproc)}
TAG=${TAG:-pilot}
LANES=${LANES:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/coplanar_limit_full}
OUT=${OUT:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/tube_features_$TAG}
EXTRA=${EXTRA:-}
cd "$ROOT" || exit 1
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no $PY"; exit 2; }
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd -- pip install -e <rmhd_final>"; exit 2; }
. "$RMHD/scripts/kernels.env"
ls "$LANES"/shard_*.npz >/dev/null 2>&1 || { echo "PREFLIGHT FAILED: no shards in $LANES"; exit 2; }
[ -e "$OUT" ] && { echo "PREFLIGHT FAILED: $OUT exists -- pick another TAG"; exit 2; }
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: N=$N per group, $NW workers, $LANES -> $OUT"; exit 0; }

export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- scripts)" ] || echo '+uncommitted')  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== N=$N per group  workers=$NW  $LANES -> $OUT"
pids=()
for ((i=0; i<NW; i++)); do
    "$PY" -u scripts/tube_features.py "$LANES" --n "$N" --shard "$i" --nshards "$NW" \
        --out "$OUT/shard_$(printf '%03d' $i).npz" $EXTRA \
        > "$OUT/logs/w$(printf '%03d' $i).log" 2>&1 &
    pids+=($!)
done
nfail=0
for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
echo "== $(date) done: $(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l)/$NW shards, $nfail workers failed"
"$PY" scripts/tube_features.py --summarise "$OUT" | tee "$OUT/summary.txt"
[ "$nfail" = 0 ]
