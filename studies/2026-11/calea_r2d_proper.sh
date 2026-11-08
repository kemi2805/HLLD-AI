#!/bin/bash
#SBATCH --job-name=r2d-proper
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The four-quadrant warm-start retrain done properly (user, 2026-11-03: "fix
# the validation split and retrain properly").  The audit (r2d_audit.py) found
# the trainer validating on a random slice of a file holding each problem
# 2-20 times, the proxy drawn from the test windows, no gap between training
# and test times, iteration 3 started from a checkpoint the test faces had
# picked, and 39 stored rows that are not seven-wave solutions.  Here:
#   1. r2d_split2.py: one row per problem, verified seven-wave solutions only,
#      TEST t = 0.175 / 0.325 +- 0.0125 (as before), VAL t = 0.25 +- 0.0125,
#      0.00625 buffers dropped, perturbed faces only next to TRAIN problems;
#   2. two fine-tunes of production's network (rotor_ft: the clean lineage),
#      validation loss and proxy both on VAL (rmhd_final-valsplit = 9ca5a4b +
#      patch_val_split.py, --val-data; uncommitted): p_run on the run's
#      problems, p_pert with the perturbed faces as well;
#   3. the choice on VAL only (paired one-attempt test, r2d_choose2.py);
#   4. the test, once: production, ft1, ft3, the four new checkpoints and the
#      neutral guess on TEST.
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain
R=$B/HLLD/results/r2d_retrain
W=$R/proper
WT=$B/rmhd_final-valsplit
PY=$HOME/venv/rmhd/bin/python
INIT=$B/rmhd_final/data/ml_guess_rotor_ft.pt
mkdir -p "$W"
cd "$WT" || exit 1
export PYTHONPATH=$WT
. scripts/kernels.env
echo "== $(date) $(hostname)  rmhd_final-valsplit $(git rev-parse --short HEAD), $(git diff --shortstat)"
"$PY" -c 'import inspect, rmhd, rmhd.ml_guess as m; print("rmhd from", rmhd.__file__); assert "val_path" in inspect.signature(m.train_from_dataset).parameters' || exit 2

echo "== 1. split  $(date)"
OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/r2d_split2.py" "$R" "$W" --buffer 0.00625 || exit 3

echo "== 2. train  $(date)"
pids=()
for v in run run_pert; do
    OMP_NUM_THREADS=32 NUMBA_NUM_THREADS=16 "$PY" -u scripts/train_variant.py --data "$W/train_$v.npz" \
        --val-data "$W/val.npz" --out "$W/ml_guess_r2d_p_$v.pt" --init "$INIT" --lr 2e-4 --epochs 150 \
        --batch 256 --seed 42 --gamma 1.6666666666666667 --threads 32 --device cpu \
        --proxy-evalset "$W/proxy_val.npz" --proxy-every 10 > "$W/train_$v.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do
    wait "$p" || { echo "a training failed (pid $p)"; tail -n 5 "$W"/train_*.log; exit 4; }
done
tail -n 4 "$W"/train_*.log

NETS=(p_run="$W/ml_guess_r2d_p_run.pt" p_run_proxy="$W/ml_guess_r2d_p_run_proxy.pt"
      p_pert="$W/ml_guess_r2d_p_run_pert.pt" p_pert_proxy="$W/ml_guess_r2d_p_run_pert_proxy.pt")
echo "== 3. the choice, on VAL only  $(date)"
HLLD_DIR=$B/HLLD OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/r2d_eval_seeds.py" \
    "$W/val.npz" "$W/eval_val.npz" 6000 "${NETS[@]}" || exit 5
"$PY" "$S/r2d_choose2.py" "$W/eval_val.npz" "$W/chosen.txt" "${NETS[@]}" || exit 6
read -r CH CHP < "$W/chosen.txt"

echo "== 4. the test, once  $(date)"
REF=(production="$INIT" ft1="$R/it1/ml_guess_r2d_ft1.pt" ft3="$R/it3/ml_guess_r2d_it3.pt")
HLLD_DIR=$B/HLLD OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=32 "$PY" -u "$S/r2d_eval_seeds.py" \
    "$W/test.npz" "$W/eval_test.npz" 6000 "${REF[@]}" "${NETS[@]}" || exit 7
"$PY" "$S/r2d_paired.py" "$W/eval_test.npz" "$W/test.npz" "$W/proxy_val.npz" \
    "$CH:production" "$CH:ft1" "$CH:ft3" "$CH:neutral" "ft3:production" || exit 8
echo "== $(date) done; chosen $CH = $CHP"
