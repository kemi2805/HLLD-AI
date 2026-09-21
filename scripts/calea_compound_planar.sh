#!/usr/bin/env bash
#SBATCH --job-name=compound
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# scripts/compound_planar.py on one calea node.  Submit from the ITP login
# node (`ssh itp`):
#   cd /mnt/rafast/miler/codes/rotor2d/HLLD && sbatch scripts/calea_compound_planar.sh
# Knobs: NW, TAG, LANES (coplanar_limit dir), SCREEN (tube_features screen
# dir), ROOT, EXTRA.  RMHD_SLOWSHOCK is UNSET on purpose: the experiment
# widens slow_shock_b.MARGIN, which only the numpy path reads.
set -u
ROOT=${ROOT:-/mnt/rafast/miler/codes/rotor2d/HLLD}
PY=${PY:-$HOME/venv/rmhd/bin/python}
NW=${NW:-$(nproc)}
TAG=${TAG:-full}
LANES=${LANES:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/coplanar_limit_full}
SCREEN=${SCREEN:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/tube_features_screen}
OUT=${OUT:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/compound_planar_$TAG}
EXTRA=${EXTRA:-}
cd "$ROOT" || exit 1
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no $PY"; exit 2; }
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd"; exit 2; }
. "$RMHD/scripts/kernels.env"
unset RMHD_SLOWSHOCK
for d in "$LANES" "$SCREEN"; do
    ls "$d"/shard_*.npz >/dev/null 2>&1 || { echo "PREFLIGHT FAILED: no shards in $d"; exit 2; }
done
[ -e "$OUT" ] && { echo "PREFLIGHT FAILED: $OUT exists -- pick another TAG"; exit 2; }
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: $NW workers -> $OUT"; exit 0; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- scripts)" ] || echo '+uncommitted')  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
pids=()
for ((i=0; i<NW; i++)); do
    "$PY" -u scripts/compound_planar.py "$LANES" "$SCREEN" --shard "$i" --nshards "$NW" \
        --out "$OUT/shard_$(printf '%03d' $i).npz" $EXTRA \
        > "$OUT/logs/w$(printf '%03d' $i).log" 2>&1 &
    pids+=($!)
done
nfail=0
for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
echo "== $(date) done: $(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l)/$NW shards, $nfail failed"
"$PY" scripts/compound_planar.py --summarise "$OUT" | tee "$OUT/summary.txt"
[ "$nfail" = 0 ]
