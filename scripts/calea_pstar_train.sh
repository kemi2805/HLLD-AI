#!/bin/bash
#SBATCH --job-name=pstar-train
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD-pstar/logs/slurm_%x_%j.out
# The p* networks on ONE calea node's CPUs: four scripts/pstar_train.py
# processes side by side, THREADS cores each (4 x 16 = the node's 64), each
# with its own feature set and seed.  (The user's decision of 2026-10-17:
# CPU here instead of a ~2.5-day wait for Goethe's gpu partition.)
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD-pstar
#   sbatch scripts/calea_pstar_train.sh
#
# Network and optimiser as the archived checkpoints/default-mignone.pt
# (15 or 17 -> [256,256,256,256,128,64] -> 1, SiLU + LayerNorm; Adam lr
# 3e-3, weight decay 1e-4, ReduceLROnPlateau 10 / 0.5 / 1e-5, MSE, clip 1,
# best-validation checkpoint).  What CPU training changes: BATCH, EPOCHS
# and the threads; pstar_train.py's --time-limit stops a run cleanly (best
# checkpoint and its .json written) before Slurm's limit.
#
# Knobs: RUNS ("name:version:extra args|..."), SPLITS
# (/mnt/rafast/miler/pstar/union/splits), OUTDIR (checkpoints/union),
# EPOCHS (300), BATCH (16384), LR (3e-3), THREADS (16), HOURS (22, the
# per-run --time-limit), PY, HLLD_DIR.  DRYRUN=1: preflight only.
set -u -o pipefail
BASE=/mnt/rafast/miler/codes/rotor2d
HLLD_DIR=${HLLD_DIR:-$BASE/HLLD-pstar}
PY=${PY:-$HOME/venv/rmhd/bin/python}
SPLITS=${SPLITS:-/mnt/rafast/miler/pstar/union/splits}
OUTDIR=${OUTDIR:-checkpoints/union}
EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-16384}
LR=${LR:-3e-3}
THREADS=${THREADS:-16}
HOURS=${HOURS:-22}
HIDDEN=${HIDDEN:-256,256,256,256,128,64}
RUNS=${RUNS:-"v2_s0:v2-idir-abs-bn:--seed 0|v2_s1:v2-idir-abs-bn:--seed 1|v3_s0:v3-hlld-r-invariants:--seed 0|v3_s1:v3-hlld-r-invariants:--seed 1"}
cd "$HLLD_DIR" || { echo "PREFLIGHT FAILED: no $HLLD_DIR"; exit 2; }
export PYTHONPATH=$HLLD_DIR
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no python $PY"; exit 2; }
[ -f scripts/pstar_train.py ] || { echo "PREFLIGHT FAILED: no scripts/pstar_train.py"; exit 2; }
IFS='|' read -ra RUNLIST <<< "$RUNS"
for r in "${RUNLIST[@]}"; do
    name=${r%%:*}; rest=${r#*:}; ver=${rest%%:*}
    [ -n "$name" ] && [ -n "$ver" ] && [ "$rest" != "$r" ] ||
        { echo "PREFLIGHT FAILED: run '$r' is not name:version:args"; exit 2; }
    for s in X_train.npy y_train.npy X_val.npy y_val.npy norm_stats.npz; do
        [ -e "$SPLITS/$ver/$s" ] || { echo "PREFLIGHT FAILED: missing $SPLITS/$ver/$s"; exit 2; }
    done
done
[ $((${#RUNLIST[@]} * THREADS)) -le 64 ] ||
    { echo "PREFLIGHT FAILED: ${#RUNLIST[@]} runs x $THREADS threads > 64 cores"; exit 2; }
mkdir -p "$OUTDIR" && echo ok > "$OUTDIR/.write_test" && rm -f "$OUTDIR/.write_test" ||
    { echo "PREFLIGHT FAILED: cannot write $OUTDIR"; exit 2; }
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: ${#RUNLIST[@]} runs x $THREADS threads, epochs=$EPOCHS batch=$BATCH lr=$LR hours=$HOURS splits=$SPLITS"; exit 0; }

mkdir -p logs
echo "== $(date)  host=$(hostname)  HLLD $(git rev-parse --short HEAD) ($(git branch --show-current))  epochs=$EPOCHS batch=$BATCH lr=$LR threads=$THREADS hidden=$HIDDEN"
"$PY" -c "import numpy, torch, src; print('numpy', numpy.__version__, 'torch', torch.__version__, torch.get_num_threads(), 'threads; src', src.__file__)"
pids=""; names=""
for r in "${RUNLIST[@]}"; do
    name=${r%%:*}; rest=${r#*:}; ver=${rest%%:*}; args=${rest#*:}
    [ "$args" = "$rest" ] && args=""
    log="$OUTDIR/train_${SLURM_JOB_ID:-0}_${name}.log"
    echo "  $name ($ver; $args) -> $OUTDIR/$name.pt  log $log"
    "$PY" -u scripts/pstar_train.py --split "$SPLITS/$ver" --out "$OUTDIR/$name.pt" \
        --device cpu --threads "$THREADS" --epochs "$EPOCHS" --batch "$BATCH" \
        --lr "$LR" --hidden "$HIDDEN" --time-limit "$HOURS" $args > "$log" 2>&1 &
    pids="$pids $!"; names="$names $name"
done
# each child's own status: a bare `wait` returns 0 even when every child died
rc=0; k=0
set -- $names
for pid in $pids; do
    k=$((k+1)); eval "nm=\${$k}"
    if wait "$pid"; then echo "RUN $nm ok: $(tail -n 1 "$OUTDIR/train_${SLURM_JOB_ID:-0}_${nm}.log")"
    else st=$?; rc=1
         echo "RUN $nm FAILED (exit $st) -- tail of its log:"
         tail -n 5 "$OUTDIR/train_${SLURM_JOB_ID:-0}_${nm}.log" | sed "s/^/    /"
    fi
done
echo "== $(date)  rc=$rc"
ls -la "$OUTDIR"
exit $rc
