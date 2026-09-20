#!/usr/bin/env bash
#SBATCH --job-name=coplanar
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The eps-regularised coplanar limit (scripts/coplanar_limit.py) on one calea
# node.  Submit from iota:
#   cd /mnt/rafast/miler/codes/rotor2d/HLLD
#   N=64 TAG=pilot sbatch scripts/calea_coplanar_limit.sh    # the pilot: 64 + 64 lanes
#   sbatch scripts/calea_coplanar_limit.sh                   # all 1280 + 1280
#
# Knobs: N (lanes per group, a fixed random subset; 0 = all), NW (worker
# processes, default the cores Slurm granted), TAG (output dir suffix),
# BRUTE (the brute-force run whose lanes are reused, so the two pair lane by
# lane), ROOT (checkout to run; a scratch worktree for an uncommitted script),
# EXTRA (passed through, e.g. "--eps 1e-1 1e-2 1e-3").  The warm start is
# production's: RMHD_ML_CKPT / RMHD_ML_CKPTS as calea_rotor_exact.sh sets them.
set -u
ROOT=${ROOT:-/mnt/rafast/miler/codes/rotor2d/HLLD}
PY=${PY:-$HOME/venv/rmhd/bin/python}
N=${N:-0}
NW=${NW:-$(nproc)}
TAG=${TAG:-full}
BRUTE=${BRUTE:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/brute_hlld}
OUT=${OUT:-/mnt/rafast/miler/codes/rotor2d/HLLD/results/coplanar_limit_$TAG}
EXTRA=${EXTRA:-}
cd "$ROOT" || exit 1
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no $PY"; exit 2; }
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd -- pip install -e <rmhd_final>"; exit 2; }
. "$RMHD/scripts/kernels.env"
export RMHD_ML_CKPT=${RMHD_ML_CKPT:-data/ml_guess_rotor_ft.pt}
export RMHD_ML_CKPTS=${RMHD_ML_CKPTS-data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt}
for c in $RMHD_ML_CKPT ${RMHD_ML_CKPTS//,/ }; do
    [ -f "$RMHD/$c" ] || { echo "PREFLIGHT FAILED: no checkpoint $RMHD/$c"; exit 2; }
done
ls "$BRUTE"/shard_*.npz >/dev/null 2>&1 || { echo "PREFLIGHT FAILED: no shards in $BRUTE"; exit 2; }
[ -e "$OUT" ] && { echo "PREFLIGHT FAILED: $OUT exists -- pick another TAG"; exit 2; }
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: N=$N per group, $NW workers, $BRUTE -> $OUT"; exit 0; }

export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/logs"
echo "== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)$([ -z "$(git status --porcelain -- scripts)" ] || echo '+uncommitted')  rmhd $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== N=$N per group  workers=$NW  $BRUTE -> $OUT  ckpt=$RMHD_ML_CKPT extra=$RMHD_ML_CKPTS"
echo "== kernels: FAN=$RMHD_FAN ALFVEN=$RMHD_ALFVEN SLOWSHOCK=$RMHD_SLOWSHOCK SHOCK=$RMHD_SHOCK"
pids=()
for ((i=0; i<NW; i++)); do
    "$PY" -u scripts/coplanar_limit.py "$BRUTE" --n "$N" --shard "$i" --nshards "$NW" \
        --out "$OUT/shard_$(printf '%03d' $i).npz" $EXTRA \
        > "$OUT/logs/w$(printf '%03d' $i).log" 2>&1 &
    pids+=($!)
done
# per-pid: a bare `wait` returns 0 however many workers crashed
nfail=0
for p in "${pids[@]}"; do wait "$p" || nfail=$((nfail + 1)); done
echo "== $(date) done: $(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l)/$NW shards, $nfail workers failed"
"$PY" scripts/coplanar_limit.py --summarise "$OUT" | tee "$OUT/summary.txt"
[ "$nfail" = 0 ]
