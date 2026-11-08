#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The planar dataset (user 2026-11-06: "yes, start the dataset job and the
# one-call version"; the planar warm-start network): the faces a run's Newton
# refused (pop_refused.py: its unsolved rows + its entropy-falling seven-wave
# answers) re-solved with the full chain -- the loss fixes, the gate, the
# rotation re-solve, the planar seven-wave rung after the five-wave solver in
# one call (RMHD_COPLANAR7=late, _CALLS=1: the same answers as staged), the
# batched starts -- with every rung's answers harvested (RMHD_HARVEST_RESCUES):
# source 0-3 rows with each side's slow-wave steering, the rotations and the
# stage.  Production networks (rotor_ft + s42/s47), pool 32x2.
#   sbatch --job-name=pds-r2d64 --export=ALL,POP=pop_r2d_64_exact_ctl1102.npz calea_planar_dataset.sh
# rmhd rmhd_final-loose10, HLLD HLLD-fix11 (c7_patches/setup_onecall.sh); uncommitted.
set -u
: "${POP:?}"
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/planar_ml
RMHD=$B/rmhd_final-loose10
export PYTHONPATH=$RMHD HLLD_DIR=$B/HLLD-fix11
PY=$HOME/venv/rmhd/bin/python
OUT=$B/HLLD/results/planar_ml/$(basename "$POP" .npz | sed 's/^pop_//')
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE RMHD_COPLANAR7_P
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1
export RMHD_PLANAR4_ROT=1 RMHD_COPLANAR7=late RMHD_COPLANAR7_CALLS=1 RMHD_PLANAR4_STAR_BATCH=1 RMHD_HARVEST_RESCUES=1
export RMHD_ML_CKPT=$RMHD/data/ml_guess_rotor_ft.pt
export RMHD_ML_CKPTS=$RMHD/data/ml_guess_rotor_big_s42.pt,$RMHD/data/ml_guess_rotor_big_s47.pt
. "$RMHD/scripts/kernels.env"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=2 TORCH_THREADS=8 RMHD_POOL=32 RMHD_POOL_THREADS=2 RMHD_POOL_CHUNKS=32
cd "$HLLD_DIR" || exit 1
grep -q _HARVEST_RESCUES src/physics/exact_flux.py || { echo "PREFLIGHT FAILED: HLLD patch"; exit 2; }
grep -q RMHD_COPLANAR7_CALLS "$RMHD/rmhd/batched/coplanar7_b.py" || { echo "PREFLIGHT FAILED: rmhd patch"; exit 2; }
[ -f "$S/$POP" ] || { echo "PREFLIGHT FAILED: no $S/$POP"; exit 2; }
mkdir -p "$OUT"
echo "== $(date) $(hostname) $POP -> $OUT: HLLD $(git rev-parse --short HEAD)+patch, rmhd $(git -C $RMHD rev-parse --short HEAD)+patch"
"$PY" -u "$S/build_dataset.py" "$S/$POP" "$OUT" --chunk 4096 2>&1 | grep --line-buffered -v -i "warn\|W = xp.sqrt" | tee -a "$OUT/build.log"
echo "== $(date) exit ${PIPESTATUS[0]}"
