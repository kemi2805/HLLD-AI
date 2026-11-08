#!/bin/bash
#SBATCH --job-name=ml-proper
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The proper retrain of the warm-start network for the rotor and the
# Orszag-Tang (user, 2026-11-03: "Please also train the networks with the fixes
# we did ... for the rotor and orszag tang"): the four-quadrant protocol of
# calea_r2d_proper.sh (job 77245) on each problem's 64^2 harvest with today's
# solver (<short>_64_exact_ssretry, 10-29/30: Alfven closed form and the
# slow-shock retry on):
#   1. the solved faces with their times (harvest_to_dataset --keep-tags);
#   2. split_proper.py phase 1: one row per problem (to ~1e-10), verified
#      seven-wave solutions only, TEST / VAL time blocks scaled to t_end with
#      buffers, complex x10 in training only;
#   3. 100k perturbed faces drawn from the training file, solved through the
#      production path (production's networks); phase 2 keeps those whose
#      nearest run problem is a TRAIN problem;
#   4. two fine-tunes of production's rotor_ft at the problem's gamma (the
#      Orszag-Tang runs at 4/3), validation loss and proxy on VAL
#      (rmhd_final-valsplit = 9ca5a4b + patch_val_split.py, uncommitted):
#      p_run on the run's problems, p_pert with the perturbed faces as well;
#   5. the choice on VAL only (r2d_choose2.py);
#   6. the test, once, on TEST: production, the two four-quadrant networks
#      (ft3, and q4_proper = the clean proper p_pert), the four new
#      checkpoints and the neutral guess.
#   ssh itp; sbatch --job-name=ml-proper-rotor --export=ALL,PROB=rotor calea_proper_2d.sh
set -u
: "${PROB:?sbatch --export=ALL,PROB=rotor|ot}"
case "$PROB" in
    rotor) RUN=rotor_64_exact_ssretry; TEND=0.4 ;;
    ot)    RUN=ot_64_exact_ssretry;    TEND=1.0 ;;
    *) echo "PROB must be rotor or ot"; exit 2 ;;
esac
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain
R=$B/HLLD/results
Q=$R/r2d_retrain
W=$R/ml_retrain/$PROB
WT=$B/rmhd_final-valsplit
HL=$B/HLLD
PY=$HOME/venv/rmhd/bin/python
D=$B/rmhd_final/data
INIT=$D/ml_guess_rotor_ft.pt
mkdir -p "$W"
cd "$WT" || exit 1
export PYTHONPATH=$WT
unset RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED
. scripts/kernels.env
echo "== $(date) $(hostname)  $PROB: $RUN, t_end $TEND"
echo "== rmhd_final-valsplit $(git rev-parse --short HEAD), $(git diff --shortstat); HLLD $(git -C "$HL" rev-parse --short HEAD)"
"$PY" -c 'import inspect, rmhd, rmhd.ml_guess as m; print("rmhd from", rmhd.__file__); assert "val_path" in inspect.signature(m.train_from_dataset).parameters' || exit 2

if [ "$PROB" = rotor ]; then
    # split_proper.py at bit-for-bit deduplication and t_end 0.4 must reproduce
    # r2d_split2.py's four-quadrant split of job 77245 (same files, same rows)
    echo "== 0. split_proper.py on the four-quadrant data (--bits 53) against job 77245's split  $(date)"
    C=$R/ml_retrain/q4_check
    OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/split_proper.py" "$Q/it1/all.npz" "$C" --tend 0.4 \
        --bits 53 > "$C.log" 2>&1 && "$PY" "$S/check_split_same.py" "$C" "$Q/proper" && rm -rf "$C"
fi

echo "== 1. dataset  $(date)"
if [ ! -f "$W/all.npz" ]; then
    OMP_NUM_THREADS=8 "$PY" scripts/harvest_to_dataset.py "$R/$RUN/harvest" --out "$W/all.npz" \
        --keep-tags --hold-out-t 99 || exit 3
fi
G=$("$PY" -c "import numpy as np; print(repr(float(np.load('$W/all.npz')['gamma'])))") || exit 3
echo "   gamma $G"

echo "== 2. split, phase 1  $(date)"
OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/split_proper.py" "$W/all.npz" "$W" --tend "$TEND" || exit 4

echo "== 3. perturbed faces  $(date)"
( cd "$HL" && export HLLD_DIR=$HL RMHD_ML_CKPT=$INIT \
      RMHD_ML_CKPTS=$D/ml_guess_rotor_big_s42.pt,$D/ml_guess_rotor_big_s47.pt \
      RMHD_POOL=63 RMHD_POOL_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 &&
  "$PY" -u "$S/r2d_resolve.py" perturb "$W/train_run.npz" "$W/perturb" --n 100000 --sigma 0.05 ) || exit 5
OMP_NUM_THREADS=8 "$PY" scripts/harvest_to_dataset.py "$W/perturb/harvest" --out "$W/perturb.npz" || exit 5
echo "== split, phase 2  $(date)"
OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/split_proper.py" "$W/all.npz" "$W" --tend "$TEND" \
    --perturb "$W/perturb.npz" > "$W/split_phase2.log" 2>&1 || { tail -n 20 "$W/split_phase2.log"; exit 6; }
"$PY" -c "import json; print(json.dumps(json.load(open('$W/split.json'))['perturbed']))"

echo "== 4. train  $(date)"
pids=()
for v in run run_pert; do
    OMP_NUM_THREADS=32 NUMBA_NUM_THREADS=16 "$PY" -u scripts/train_variant.py --data "$W/train_$v.npz" \
        --val-data "$W/val.npz" --out "$W/ml_guess_${PROB}_p_$v.pt" --init "$INIT" --lr 2e-4 --epochs 150 \
        --batch 256 --seed 42 --gamma "$G" --threads 32 --device cpu \
        --proxy-evalset "$W/proxy_val.npz" --proxy-every 10 > "$W/train_$v.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do
    wait "$p" || { echo "a training failed (pid $p)"; tail -n 5 "$W"/train_*.log; exit 7; }
done
tail -n 4 "$W"/train_*.log

NETS=(p_run="$W/ml_guess_${PROB}_p_run.pt" p_run_proxy="$W/ml_guess_${PROB}_p_run_proxy.pt"
      p_pert="$W/ml_guess_${PROB}_p_run_pert.pt" p_pert_proxy="$W/ml_guess_${PROB}_p_run_pert_proxy.pt")
echo "== 5. the choice, on VAL only  $(date)"
HLLD_DIR=$HL OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/r2d_eval_seeds.py" \
    "$W/val.npz" "$W/eval_val.npz" 6000 "${NETS[@]}" || exit 8
"$PY" "$S/r2d_choose2.py" "$W/eval_val.npz" "$W/chosen.txt" "${NETS[@]}" || exit 9
read -r CH CHP < "$W/chosen.txt"

echo "== 6. the test, once  $(date)"
REF=(production="$INIT" q4_ft3="$Q/it3/ml_guess_r2d_it3.pt" q4_proper="$Q/proper/ml_guess_r2d_p_run_pert.pt")
HLLD_DIR=$HL OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/r2d_eval_seeds.py" \
    "$W/test.npz" "$W/eval_test.npz" 6000 "${REF[@]}" "${NETS[@]}" || exit 10
"$PY" "$S/r2d_paired.py" "$W/eval_test.npz" "$W/test.npz" "$W/proxy_val.npz" \
    "$CH:production" "$CH:q4_proper" "$CH:q4_ft3" "$CH:neutral" "q4_proper:production" "production:neutral" || exit 11
echo "== $(date) done; chosen $CH = $CHP"
