#!/usr/bin/env bash
#SBATCH --job-name=census
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# scripts/snapshot_census.py on one calea node: every attempted interface of
# a rotor run's snapshots, solved with today's code and tube-tested.
# Submit from the ITP login node (`ssh itp`):
#   cd /mnt/rafast/miler/codes/rotor2d/HLLD && sbatch scripts/calea_snapshot_census.sh
# Knobs: RUN (a rotor run dir with snap_*.npz), TAG, NW, ROOT, EXTRA (e.g.
# "--times 0.2,0.4 --ncells 512").  Production kernels and warm start.
# NSHARDS > NW runs only shards 0..NW-1 of NSHARDS: a systematic subsample
# of NW/NSHARDS of the lanes (NW=64 NSHARDS=1280 EXTRA="--ncells 512" is the
# tube-resolution check of plan F step 2: 5% of the lanes at 512 + 1024).
set -u
ROOT=${ROOT:-/mnt/rafast/miler/codes/rotor2d/HLLD}
PY=${PY:-$HOME/venv/rmhd/bin/python}
NW=${NW:-$(nproc)}
NSHARDS=${NSHARDS:-$NW}
RUN=${RUN:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/rotor_64_exact}
TAG=${TAG:-64}
OUT=${OUT:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/snapshot_census_$TAG}
EXTRA=${EXTRA:-}
cd "$ROOT" || exit 1
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no $PY"; exit 2; }
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd"; exit 2; }
. "$RMHD/scripts/kernels.env"
export RMHD_ML_CKPT=${RMHD_ML_CKPT:-data/ml_guess_rotor_ft.pt}
export RMHD_ML_CKPTS=${RMHD_ML_CKPTS-data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt}
ls "$RUN"/snap_*.npz >/dev/null 2>&1 || { echo "PREFLIGHT FAILED: no snapshots in $RUN"; exit 2; }
[ -e "$OUT" ] && { echo "PREFLIGHT FAILED: $OUT exists -- pick another TAG"; exit 2; }
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: $RUN -> $OUT, $NW workers"; exit 0; }
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- scripts)" ] || echo '+uncommitted')  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== $RUN -> $OUT  workers=$NW of $NSHARDS shards  $EXTRA"
pids=()
for ((i=0; i<NW; i++)); do
    "$PY" -u scripts/snapshot_census.py "$RUN" --shard "$i" --nshards "$NSHARDS" \
        --out "$OUT/shard_$(printf '%03d' $i).npz" $EXTRA \
        > "$OUT/logs/w$(printf '%03d' $i).log" 2>&1 &
    pids+=($!)
done
nfail=0
for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
echo "== $(date) done: $(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l)/$NW shards, $nfail failed"
"$PY" scripts/snapshot_census.py --summarise "$OUT" | tee "$OUT/summary.txt"
[ "$nfail" = 0 ]
