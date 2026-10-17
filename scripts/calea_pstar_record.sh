#!/bin/bash
#SBATCH --job-name=pstar-record
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD-pstar/logs/slurm_%x_%j.out
# Real interface states for the p* network's test sets: the plain-HLLD rotor
# (256^2, t = 0.4) and Orszag-Tang (128^2, t = 1.0) runs of
# docs/rotor_envelope.md and docs/ot_envelope.md, re-run with the envelope
# recorder keeping the FULL L/R primitive states of every sampled face
# (run_2d.py --envelope-states).  The September envelope files hold eleven
# summary magnitudes per face, which cannot be re-solved.
#
# --envelope-every is 31, not the September 30: an SSP-RK3 step makes six
# sweep calls (3 stages x 2 axes), so every=30 recorded call 1 mod 30 = the
# first stage's x-sweep only, and the September envelopes never saw a
# y-sweep.  31 cycles through all six.  3000 faces per recorded call as then.
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD-pstar
#   sbatch scripts/calea_pstar_record.sh
#
# Both runs on this one node, side by side (TORCH_THREADS each).  Knobs:
# EVERY (31), SAMPLES (3000), TORCH_THREADS (16), OUT (results/pstar_rec),
# PY, HLLD_DIR, ONLY (rotor | ot: one of the two).  DRYRUN=1: preflight only.
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
HLLD_DIR=${HLLD_DIR:-$BASE/HLLD-pstar}
PY=${PY:-$HOME/venv/rmhd/bin/python}
EVERY=${EVERY:-31}
SAMPLES=${SAMPLES:-3000}
OUT=${OUT:-results/pstar_rec}
ONLY=${ONLY:-}
cd "$HLLD_DIR" || { echo "PREFLIGHT FAILED: no $HLLD_DIR"; exit 2; }
# The venv has the PRODUCTION checkout's src installed editable; without this
# every import below would silently run that code, not this branch's.
export PYTHONPATH=$HLLD_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TORCH_THREADS=${TORCH_THREADS:-16}

[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no python $PY"; exit 2; }
[ $((EVERY % 2)) = 1 ] || { echo "PREFLIGHT FAILED: EVERY=$EVERY is even (x-sweeps only)"; exit 2; }
grep -q -- "--envelope-states" scripts/run_2d.py ||
    { echo "PREFLIGHT FAILED: this run_2d.py has no --envelope-states"; exit 2; }
# (the login node has no torch: on kappa the import check is skipped)
if "$PY" -c "import torch" 2>/dev/null; then
    where=$("$PY" -c "import src, os; print(os.path.dirname(os.path.dirname(os.path.abspath(src.__file__))))")
    [ "$where" = "$(pwd -P)" ] || [ "$where" = "$HLLD_DIR" ] ||
        { echo "PREFLIGHT FAILED: src resolves to $where, not $HLLD_DIR"; exit 2; }
fi
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: every=$EVERY samples=$SAMPLES out=$OUT threads=$TORCH_THREADS"; exit 0; }

mkdir -p logs "$OUT"
echo "== $(date)  host=$(hostname)  HLLD $(git rev-parse --short HEAD) ($(git branch --show-current))  every=$EVERY samples=$SAMPLES"
"$PY" -c "import numpy, torch, src; print('numpy', numpy.__version__, 'torch', torch.__version__, 'src', src.__file__)"

run() {   # run <tag> <problem> <n> <tend>
    "$PY" -u scripts/run_2d.py --problem "$2" --n "$3" --tend "$4" --solver hlld \
        --limiter mc --cfl 0.25 --nsnap 2 --out "$OUT/run_$1" \
        --envelope "$OUT/$1_states.npz" --envelope-every "$EVERY" \
        --envelope-samples "$SAMPLES" --envelope-states \
        > "$OUT/$1.log" 2>&1
}
pids=""; tags=""
if [ -z "$ONLY" ] || [ "$ONLY" = rotor ]; then
    run rotor_256 rotor 256 0.4 & pids="$pids $!"; tags="$tags rotor_256"
fi
if [ -z "$ONLY" ] || [ "$ONLY" = ot ]; then
    run ot_128 orszag_tang 128 1.0 & pids="$pids $!"; tags="$tags ot_128"
fi
# a bare `wait` returns 0 even when every child died: keep each status
rc=0; set -- $tags
for pid in $pids; do
    t=$1; shift
    if wait "$pid"; then echo "== $(date)  $t done"
    else st=$?; rc=1; echo "== $(date)  $t FAILED (exit $st):"; tail -5 "$OUT/$t.log"; fi
done
ls -la "$OUT"
exit $rc
