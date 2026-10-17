#!/bin/bash
#SBATCH --job-name=pstar-eval
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD-pstar/logs/slurm_%x_%j.out
# Score the p* networks against today's HLLD (scripts/pstar_evaluate.py) on
#   synthetic  the new test split ($DATA/splits/test_rows.npz)
#   rotor      the recorded rotor faces ($DATA/real/rotor_256_rows.npz)
#   ot         the recorded Orszag-Tang faces ($DATA/real/ot_128_rows.npz)
# and run the 1D tubes TUBES with `hlld_ai` against `hlld`
# (scripts/pstar_tube_compare.py).
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD-pstar
#   MODELS="archived=checkpoints/archived/default-mignone.pt,checkpoints/archived/norm_stats.npz new_v2=..." \
#       sbatch scripts/calea_pstar_eval.sh
#
# Knobs: MODELS (space-separated NAME=CKPT,STATS[,hidden=..]; default: the
# archived checkpoint and every $CKDIR/*.pt with its _norm_stats.npz), CKDIR
# (checkpoints/union), DATA (/mnt/rafast/miler/pstar/union), OUT
# (results/pstar_eval), TUBES (configs/mignone/st1.yaml configs/mignone/st2.yaml),
# STAGES ("eval tubes"), WORKERS (60), MAX_ROWS (all), PY, HLLD_DIR.
set -u -o pipefail
BASE=/mnt/rafast/miler/codes/rotor2d
HLLD_DIR=${HLLD_DIR:-$BASE/HLLD-pstar}
PY=${PY:-$HOME/venv/rmhd/bin/python}
DATA=${DATA:-/mnt/rafast/miler/pstar/union}
CKDIR=${CKDIR:-checkpoints/union}
OUT=${OUT:-results/pstar_eval}
TUBES=${TUBES:-"configs/mignone/st1.yaml configs/mignone/st2.yaml"}
STAGES=${STAGES:-"eval tubes"}
WORKERS=${WORKERS:-60}
MAX_ROWS=${MAX_ROWS:-}
cd "$HLLD_DIR" || { echo "PREFLIGHT FAILED: no $HLLD_DIR"; exit 2; }
export PYTHONPATH=$HLLD_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

has() { case " $STAGES " in *" $1 "*) return 0 ;; esac; return 1; }
if [ -z "${MODELS:-}" ]; then
    MODELS="archived=checkpoints/archived/default-mignone.pt,checkpoints/archived/norm_stats.npz"
    for c in "$CKDIR"/*.pt; do
        [ -e "$c" ] || continue
        MODELS="$MODELS $(basename "${c%.pt}")=$c,${c%.pt}_norm_stats.npz"
    done
fi
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no python $PY"; exit 2; }
for m in $MODELS; do
    spec=${m#*=}; ck=${spec%%,*}; rest=${spec#*,}; st=${rest%%,*}
    for f in "$ck" "$st"; do [ -f "$f" ] || { echo "PREFLIGHT FAILED: model ${m%%=*}: missing $f"; exit 2; }; done
done
TESTSETS=""
for ts in synthetic=$DATA/splits/test_rows.npz rotor=$DATA/real/rotor_256_rows.npz \
          ot=$DATA/real/ot_128_rows.npz; do
    [ -f "${ts#*=}" ] || { echo "PREFLIGHT FAILED: missing test set ${ts#*=}"; exit 2; }
    TESTSETS="$TESTSETS --testset $ts"
done
for t in $TUBES; do [ -f "$t" ] || { echo "PREFLIGHT FAILED: missing $t"; exit 2; }; done
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: models: $(for m in $MODELS; do printf '%s ' "${m%%=*}"; done) out=$OUT"; exit 0; }

mkdir -p "$OUT"
echo "== $(date)  host=$(hostname)  HLLD $(git rev-parse --short HEAD)  models: $(for m in $MODELS; do printf '%s ' "${m%%=*}"; done)"
rc=0
MARGS=""; for m in $MODELS; do MARGS="$MARGS --model $m"; done
if has eval; then
    echo "== eval  $(date +%T)"
    "$PY" -u scripts/pstar_evaluate.py $MARGS $TESTSETS --out "$OUT/eval.json" \
        --workers "$WORKERS" --chunk 10000 --save-rows "$OUT/rows" \
        ${MAX_ROWS:+--max-rows "$MAX_ROWS"} || rc=1
fi
if has tubes; then
    echo "== tubes  $(date +%T)"
    P=""
    for t in $TUBES; do
        tag=$(basename "${t%.yaml}")
        OMP_NUM_THREADS=4 "$PY" -u scripts/pstar_tube_compare.py --config "$t" $MARGS \
            --out "$OUT/tubes" > "$OUT/tube_$tag.log" 2>&1 & P="$P $!"
    done
    for p in $P; do wait "$p" || { echo "  tube process $p FAILED"; rc=1; }; done
    for t in $TUBES; do tag=$(basename "${t%.yaml}"); echo "-- $tag"; grep -v "^  step\|Starting\|Done" "$OUT/tube_$tag.log" | tail -n 12; done
fi
echo "== $(date)  rc=$rc"
exit $rc
