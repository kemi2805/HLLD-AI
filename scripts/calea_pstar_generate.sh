#!/bin/bash
#SBATCH --job-name=pstar-gen
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD-pstar/logs/slurm_%x_%j.out
# The p* training data on one calea node, in stages:
#
#   gen    CHUNKS chunks of configs/pstar-union.yaml (rows_per_chunk each,
#          gamma by chunk parity) -> $DATA/shards/shard_*.npz, every row kept
#          with its flags.  Resumable: an existing shard is skipped.
#   build  valid rows -> $DATA/splits (train/val/test per feature version,
#          asinh-standardised, feature_version stamped; test_rows.npz,
#          invalid_rows.npz, summary.json)
#   real   the recorded rotor / OT faces (calea_pstar_record.sh) -> labelled
#          $DATA/real/{rotor_256,ot_128}_rows.npz, all faces kept, flagged
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD-pstar
#   sbatch scripts/calea_pstar_generate.sh
#   STAGES=real sbatch --time=1:00:00 scripts/calea_pstar_generate.sh
#
# Knobs: STAGES ("gen build real"), DATA (/mnt/rafast/miler/pstar/union),
# CHUNKS (0-63), WORKERS (60), TRANSFORM (asinh), REC (results/pstar_rec),
# PY, HLLD_DIR.  DRYRUN=1: preflight only.
set -u -o pipefail
BASE=/mnt/rafast/miler/codes/rotor2d
HLLD_DIR=${HLLD_DIR:-$BASE/HLLD-pstar}
PY=${PY:-$HOME/venv/rmhd/bin/python}
DATA=${DATA:-/mnt/rafast/miler/pstar/union}
STAGES=${STAGES:-"gen build real"}
CHUNKS=${CHUNKS:-0-63}
WORKERS=${WORKERS:-60}
TRANSFORM=${TRANSFORM:-asinh}
REC=${REC:-results/pstar_rec}
cd "$HLLD_DIR" || { echo "PREFLIGHT FAILED: no $HLLD_DIR"; exit 2; }
export PYTHONPATH=$HLLD_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

has() { case " $STAGES " in *" $1 "*) return 0 ;; esac; return 1; }
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no python $PY"; exit 2; }
for f in configs/pstar-union.yaml scripts/pstar_make_dataset.py \
         scripts/pstar_build_splits.py scripts/pstar_label_states.py; do
    [ -f "$f" ] || { echo "PREFLIGHT FAILED: missing $f"; exit 2; }
done
if has real && ! has gen; then
    for t in rotor_256 ot_128; do
        [ -f "$REC/${t}_states.npz" ] || { echo "PREFLIGHT FAILED: no $REC/${t}_states.npz"; exit 2; }
    done
fi
# CephFS can refuse writes while df looks fine: test with a write
mkdir -p "$DATA" && echo ok > "$DATA/.write_test" && rm -f "$DATA/.write_test" ||
    { echo "PREFLIGHT FAILED: cannot write $DATA"; exit 2; }
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: stages '$STAGES' data=$DATA chunks=$CHUNKS workers=$WORKERS"; exit 0; }

echo "== $(date)  host=$(hostname)  HLLD $(git rev-parse --short HEAD) ($(git branch --show-current))  stages: $STAGES  data=$DATA"
"$PY" -c "import numpy, torch, src; print('numpy', numpy.__version__, 'torch', torch.__version__, 'src', src.__file__)"
rc=0
if has gen; then
    echo "== gen  $(date +%T)"
    "$PY" -u scripts/pstar_make_dataset.py --config configs/pstar-union.yaml \
        --out "$DATA/shards" --chunks "$CHUNKS" --workers "$WORKERS" || rc=1
fi
if has build; then
    echo "== build  $(date +%T)"
    OMP_NUM_THREADS=16 "$PY" -u scripts/pstar_build_splits.py --shards "$DATA/shards" \
        --out "$DATA/splits" --transform "$TRANSFORM" || rc=1
fi
if has real; then
    echo "== real  $(date +%T)"
    mkdir -p "$DATA/real"
    for t in rotor_256 ot_128; do
        if [ -f "$REC/${t}_states.npz" ]; then
            "$PY" -u scripts/pstar_label_states.py "$REC/${t}_states.npz" \
                --out "$DATA/real/${t}_rows.npz" --workers 32 || rc=1
        else
            echo "  no $REC/${t}_states.npz yet"; rc=1
        fi
    done
fi
echo "== $(date)  rc=$rc"
du -sh "$DATA"/* 2>/dev/null
exit $rc
