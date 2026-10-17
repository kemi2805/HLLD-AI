#!/bin/bash
#SBATCH --job-name=pstar-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/pstar_train_%j.out
#SBATCH --error=logs/pstar_train_%j.err
# ==========================================================================
#  HLLD p* networks (PressureNet) on one MI210 node: one
#  scripts/pstar_train.py process per GPU, each with its own feature set
#  and arguments.  Submit from the checkout, through the preflight wrapper:
#
#    cd /work/astro/miler/codes/HLLD-pstar
#    /work/astro/miler/codes/rmhd_final/scripts/sbatch_checked.sh \
#        scripts/goethe_pstar_train.sbatch
#    # a pilot (gpu_test is for pilots only):
#    RUNS="pv2:v2-idir-abs-bn:--max-train 200000" EPOCHS=3 SPLITS=... \
#        .../sbatch_checked.sh scripts/goethe_pstar_train.sbatch \
#        -p gpu_test -t 0:30:00 --gres=gpu:1
#
#  Knobs: RUNS ("name:feature_version:extra args|...", one per GPU), SPLITS
#  (the directory pstar_build_splits.py wrote; SPLITS/<version>/ per run),
#  OUTDIR (checkpoints: OUTDIR/<name>.pt + .json + _norm_stats.npz), EPOCHS
#  (300), BATCH (32768), LR (3e-3), HIDDEN (the archived checkpoint's
#  256,256,256,256,128,64), PYTHON, PROJECT.
#
#  The node's host cores run at ~400 MHz under load (memory: Goethe gotchas),
#  so pstar_train.py keeps the whole split on the GPU and draws each epoch's
#  permutation there; the batch is large for the same reason.
# ==========================================================================
set -u
PROJECT=${PROJECT:-/work/astro/miler/codes/HLLD-pstar}
PYTHON=${PYTHON:-/work/astro/miler/python-env/rmhd-rocm/bin/python}
SPLITS=${SPLITS:-/work/astro/miler/pstar/splits_union}
OUTDIR=${OUTDIR:-/work/astro/miler/pstar/checkpoints}
EPOCHS=${EPOCHS:-300}
BATCH=${BATCH:-32768}
LR=${LR:-3e-3}
HIDDEN=${HIDDEN:-256,256,256,256,128,64}
RUNS=${RUNS:-"v2_s0:v2-idir-abs-bn:--seed 0|v2_s1:v2-idir-abs-bn:--seed 1|v3_s0:v3-hlld-r-invariants:--seed 0|v3_s1:v3-hlld-r-invariants:--seed 1"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${OMP_NUM_THREADS:-8}

# the modules exist only on the compute node; a login-node DRYRUN must not fail
if [ "${DRYRUN:-0}" = 1 ]; then
    module restore GRACE2026 >/dev/null 2>&1 || module load rocm/7.2.0 >/dev/null 2>&1 || true
else
    module restore GRACE2026 2>/dev/null || module load rocm/7.2.0
fi
cd "$PROJECT" || { echo "PREFLIGHT FAILED: no $PROJECT"; exit 2; }
# the editable `src` of another checkout must not shadow this one
export PYTHONPATH=$PROJECT

for f in "$PYTHON" scripts/pstar_train.py src/models/network.py; do
    [ -e "$f" ] || { echo "PREFLIGHT FAILED: missing $f"; exit 2; }
done
IFS='|' read -ra RUNLIST <<< "$RUNS"
for r in "${RUNLIST[@]}"; do
    name=${r%%:*}; rest=${r#*:}; ver=${rest%%:*}
    [ -n "$name" ] && [ -n "$ver" ] && [ "$rest" != "$r" ] ||
        { echo "PREFLIGHT FAILED: run '$r' is not name:version:args"; exit 2; }
    for s in X_train.npy y_train.npy X_val.npy y_val.npy norm_stats.npz; do
        [ -e "$SPLITS/$ver/$s" ] || { echo "PREFLIGHT FAILED: missing $SPLITS/$ver/$s"; exit 2; }
    done
done
"$PYTHON" -c "import torch, numpy" 2>/dev/null ||
    { echo "PREFLIGHT FAILED: $PYTHON cannot import torch/numpy"; exit 2; }
"$PYTHON" -c "import os, src.models.network as m; p = os.path.realpath(m.__file__); assert p.startswith(os.path.realpath('$PROJECT')), p; print('src from', os.path.dirname(p))" 2>/dev/null ||
    { echo "PREFLIGHT FAILED: src.models does not import from $PROJECT"; exit 2; }
echo "preflight ok: $PYTHON, ${#RUNLIST[@]} run(s): $(for r in "${RUNLIST[@]}"; do printf '%s ' "${r%%:*}"; done)  splits=$SPLITS"
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok"; exit 0; }
mkdir -p logs "$OUTDIR"

echo "=== $(date) $(hostname)  HLLD $(git rev-parse --short HEAD)  epochs=$EPOCHS batch=$BATCH lr=$LR hidden=$HIDDEN ==="
"$PYTHON" -c "import torch; print('torch', torch.__version__, torch.cuda.device_count(), 'GPU(s)')" 2>&1 | grep -v "^(null)"
i=0; pids=""; names=""
for r in "${RUNLIST[@]}"; do
    name=${r%%:*}; rest=${r#*:}; ver=${rest%%:*}; args=${rest#*:}
    [ "$args" = "$rest" ] && args=""
    log="logs/pstar_train_${SLURM_JOB_ID:-0}_${name}.log"
    echo "  GPU $i -> $name ($ver; $args) -> $OUTDIR/$name.pt  log $log"
    HIP_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i \
    "$PYTHON" -u scripts/pstar_train.py --split "$SPLITS/$ver" \
        --out "$OUTDIR/$name.pt" --device cuda --epochs "$EPOCHS" \
        --batch "$BATCH" --lr "$LR" --hidden "$HIDDEN" $args > "$log" 2>&1 &
    pids="$pids $!"; names="$names $name"
    i=$((i+1))
done
# `wait` with no argument returns 0 even when every child died (job 1770884
# reported COMPLETED on eight crashed runs): wait on each pid, keep its status
rc=0; k=0
set -- $names
for pid in $pids; do
    k=$((k+1)); eval "nm=\${$k}"
    if wait "$pid"; then echo "RUN $nm ok: $(tail -n 1 "logs/pstar_train_${SLURM_JOB_ID:-0}_${nm}.log")"
    else st=$?; rc=1
         echo "RUN $nm FAILED (exit $st) -- tail of its log:"
         tail -n 5 "logs/pstar_train_${SLURM_JOB_ID:-0}_${nm}.log" | sed "s/^/    /"
    fi
done
echo "=== $(date) done (rc=$rc) ==="
ls -la "$OUTDIR"
[ "$rc" = 0 ] || echo "JOB FAILED: at least one run crashed; its checkpoint is PARTIAL"
exit $rc
