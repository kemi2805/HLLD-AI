#!/usr/bin/env bash
#SBATCH --job-name=brute
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Brute-force the stubborn interfaces across all 64 cores of one calea node.
# Submit from iota (never ssh/nohup on calea01/02 since 2026-09-17):
#   cd /mnt/rafast/miler/codes/rotor2d/HLLD && BOX=hlld sbatch scripts/calea_brute.sh
#
#   BOX=global      scripts/calea_brute.sh  # asymmetric global field box
#   BOX=hlld        scripts/calea_brute.sh  # per-interface, centred on HLLD
#   BOX=rebal       scripts/calea_brute.sh  # global centre, budget on the field
#   BOX=rebal-hlld  scripts/calea_brute.sh  # HLLD centre, budget on the field
#
# The two differ only in where the field box is CENTRED, which is the whole
# question: measured over 24,632 known answers the true ln|Bt*| sits within
# [-1.54, +1.37] of HLLD's HLL average state (p1..p99) against [-7.53, -0.13]
# of the global ln sqrt(2 P_bar), and 25.6% of interfaces need the global box
# wider than +-3 against 1.3% for HLLD's.
set -u
ROOT=${ROOT:-/mnt/rafast/miler/codes/rotor2d/HLLD}
PY=${PY:-$HOME/venv/rmhd/bin/python}
BOX=${BOX:-hlld}
N=${N:-1280}
NW=${NW:-64}
OUT=${OUT:-$ROOT/results/brute_$BOX}
cd "$ROOT" || exit 1
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no $PY"; exit 2; }
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd -- pip install -e <rmhd_final>"; exit 2; }
. "$RMHD/scripts/kernels.env"
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -d results/rotor_64_exact/harvest ] || { echo "PREFLIGHT FAILED: no harvest"; exit 2; }
# rebal-*: the point budget moved off the pressures and onto the field.
# Measured over 1280 known answers, ln|Bt| carries 91.6% of the squared
# distance from the grid to the true answer and each pressure 4.2% (their
# true offsets have median +0.004, p99 +0.40) -- so +-1.5 x 11 points on each
# pressure was budget spent where the answer already sits.  Same grid cost,
# median seed error 0.104 -> 0.029, fraction inside the 0.10 convergence
# threshold 46.6% -> 82.9%.
case "$BOX" in
  hlld)        EXTRA="--hlld-box --blo -7 --bhi 9" ;;
  global)      EXTRA="--blo -16 --bhi 2" ;;
  rebal)       EXTRA="--blo -16 --bhi 2 --nb 300 --npres 5 --hp 0.6" ;;
  rebal-hlld)  EXTRA="--hlld-box --blo -7 --bhi 9 --bcore 0.03 --btail 0.3 --npres 5 --hp 0.6" ;;
  *) echo "BOX must be hlld, global, rebal or rebal-hlld"; exit 2 ;;
esac
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok ($BOX box, $NW workers, N=$N)"; exit 0; }
mkdir -p "$OUT/logs"
echo "== $(date) $(hostname)  BOX=$BOX  N=$N  workers=$NW"
for ((i=0; i<NW; i++)); do
    "$PY" -u scripts/brute_force_stubborn.py results/rotor_64_exact/harvest \
        --n "$N" --shard "$i" --nshards "$NW" $EXTRA \
        --out "$OUT/shard_$(printf '%03d' $i).npz" \
        > "$OUT/logs/w$(printf '%03d' $i).log" 2>&1 &
done
wait
echo "== $(date) done: $(ls "$OUT"/shard_*.npz 2>/dev/null | wc -l) shards"
