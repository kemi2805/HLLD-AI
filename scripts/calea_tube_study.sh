#!/bin/bash
#SBATCH --job-name=tube-study
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=4-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The 1D flux studies (scripts/tube_study.py) on one calea node: every run of
# an experiment is one single-core process, NPAR of them side by side, the
# most expensive first.  Each run writes $OUT/runs/<tag>.{npz,json} and its
# own log $OUT/logs/<tag>.log; this job's log has one start/done line each.
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD
#   EXPERIMENT=tubes    sbatch scripts/calea_tube_study.sh
#   EXPERIMENT=standing sbatch scripts/calea_tube_study.sh
#   EXPERIMENT=pilot OUT=results/tube_pilot sbatch --time=0:45:00 scripts/calea_tube_study.sh
#
# The 2026-10-15 runs: EXPERIMENT=all NPAR=60 RMHD_CONTINUATION=1
# RMHD_ML_CKPT=data/ml_guess_gamma53_v5.pt (the tube network and the
# continuation rung: on the pilot's tubes 89-100% of attempted faces exact,
# against 73-94% with the 2D production settings, which the _prod tasks keep).
#
# Knobs (env): EXPERIMENT (tubes | standing | all | pilot | levers), OUT
# (results/tube_study), NPAR (56 processes), PY, HLLD_DIR, RMHD_ML_CKPT /
# RMHD_ML_CKPTS (the production three by default, as calea_rotor_exact.sh).
# A finished run is skipped on resubmission (its .npz exists), so a job that
# hits the time limit can simply be submitted again.
set -u
EXPERIMENT=${EXPERIMENT:-tubes}
OUT=${OUT:-results/tube_study}
NPAR=${NPAR:-56}

BASE=/mnt/rafast/miler/codes/rotor2d
PY=${PY:-$HOME/venv/rmhd/bin/python}
cd "${HLLD_DIR:-$BASE/HLLD}" || exit 1
RMHD=$("$PY" -c 'import rmhd.paths as p; print(p.repo_root())') ||
    { echo "PREFLIGHT FAILED: $PY cannot import rmhd"; exit 2; }
export RMHD_ML_CKPT=${RMHD_ML_CKPT:-data/ml_guess_rotor_ft.pt}
export RMHD_ML_CKPTS=${RMHD_ML_CKPTS-data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt}
for c in $RMHD_ML_CKPT ${RMHD_ML_CKPTS//,/ } data/ml_guess_gamma53_v5.pt data/ml_guess_gamma53.pt; do
    case "$c" in /*) f=$c ;; *) f=$RMHD/$c ;; esac
    [ -f "$f" ] || { echo "PREFLIGHT FAILED: no checkpoint $f"; exit 2; }
done
. "$RMHD/scripts/kernels.env"
# one core per run: no thread pools anywhere, no solver worker pool
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export RMHD_POOL=0

mkdir -p logs "$OUT"
echo "== $(date)  host=$(hostname)  experiment=$EXPERIMENT  out=$OUT  npar=$NPAR"
echo "== HLLD $(git rev-parse --short HEAD) (tube_study.py $(md5sum scripts/tube_study.py | cut -c1-12))   rmhd_final $("$PY" -c 'import rmhd.paths as p; print(p.git_rev())')"
echo "== ckpt=$RMHD_ML_CKPT  extra=$RMHD_ML_CKPTS  kernels: FAN=$RMHD_FAN ALFVEN=$RMHD_ALFVEN SLOWSHOCK=$RMHD_SLOWSHOCK SHOCK=$RMHD_SHOCK"
echo "== weak flux=${RMHD_WEAK_FLUX:-(default)}  weak fallback tau=${RMHD_WEAK_FALLBACK_TAU:-(default)}  canonical=${RMHD_CANONICAL:-(default)}  continuation=${RMHD_CONTINUATION:-(default)}  (a task's --env overrides these)"

TASKS=$OUT/tasks_${SLURM_JOB_ID:-local}.txt
"$PY" scripts/tube_study.py list "$EXPERIMENT" --out "$OUT" > "$TASKS.all" ||
    { echo "PREFLIGHT FAILED: task list"; exit 2; }
# skip what an earlier job finished
: > "$TASKS"
while read -r line; do
    tag=$("$PY" - "$line" <<'PYEOF'
import sys, shlex
a = shlex.split(sys.argv[1])
g = lambda k: a[a.index(k) + 1]
suf = g("--suffix") if "--suffix" in a else ""
print("%s_%s_%s_%s%s" % (g("--case").replace(":", "_s"), g("--solver"), g("--limiter"), g("--ncells"), suf))
PYEOF
)
    [ -f "$OUT/runs/$tag.npz" ] || echo "$line" >> "$TASKS"
done < "$TASKS.all"
echo "== $(wc -l < "$TASKS") of $(wc -l < "$TASKS.all") tasks to run"

xargs -P "$NPAR" -L 1 -a "$TASKS" "$PY" -u scripts/tube_study.py
rc=$?
echo "== $(date)  xargs exit $rc  ($(ls "$OUT"/runs/*.npz 2>/dev/null | wc -l) runs on disk)"
