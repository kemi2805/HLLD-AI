#!/bin/bash
#SBATCH --job-name=r2d-retrain2
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-16:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Four-quadrant retrain loop, iteration 2 (user, 2026-11-02: "Retry the
# ledger, if new faces get solved, retrain with these faces again. And maybe
# try to create more fitting faces, and retrain on this").  NET = iteration
# 1's winner (sbatch --export=ALL,NET=<checkpoint>).
#   1. the ledger retried: every face the run lost, through the production
#      path with production's network and with NET (r2d_resolve.py lost)
#   2. 100k new four-quadrant faces near the real ones, half complex, solved
#      through the production path with NET (r2d_resolve.py perturb)
#   3. their solved rows + iteration 1's training set -> train2.npz (the
#      lost-and-now-solved rows x10)
#   4. NET fine-tuned on train2, picked by the seven-wave Newton on the proxy
#   5. the paired test on iteration 1's held-out faces
#   6. the ledger retried with iteration 2's network
# Production's retry ensemble stays (rotor_big s42 / s47), so only the
# primary network changes.
set -u
: "${NET:?sbatch --export=ALL,NET=<checkpoint>}"
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
W1=$R/r2d_retrain/it1
W=$R/r2d_retrain/it2
S=/mnt/rafast/miler/scratch/r2d_retrain
RF=$B/rmhd_final-slowfan
PY=$HOME/venv/rmhd/bin/python
H=$R/r2d_64_exact_ssretry/harvest
export PYTHONPATH=$RF HLLD_DIR=$B/HLLD-linseed
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED
export RMHD_ML_CKPTS=data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt
export RMHD_POOL=63 RMHD_POOL_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
mkdir -p "$W"
. "$RF/scripts/kernels.env"
cd "$HLLD_DIR" || exit 1
echo "== $(date) $(hostname)  NET=$NET"

echo "== 1. the ledger retried  $(date)"
RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt "$PY" -u "$S/r2d_resolve.py" lost "$H" "$W/lost_prod" || exit 2
RMHD_ML_CKPT="$NET" "$PY" -u "$S/r2d_resolve.py" lost "$H" "$W/lost_net" || exit 2

echo "== 2. new faces  $(date)"
RMHD_ML_CKPT="$NET" "$PY" -u "$S/r2d_resolve.py" perturb "$W1/train.npz" "$W/perturb" --n 100000 --sigma 0.05 || exit 3

echo "== 3. dataset  $(date)"
cd "$RF" || exit 1
to_dataset() {      # a harvest's solved rows -> trainer schema; no file when none were solved
    if compgen -G "$1/solved_*.npz" > /dev/null; then
        "$PY" scripts/harvest_to_dataset.py "$1" --out "$2" || exit 4
    else
        echo "  $1: no solved rows"; rm -f "$2"
    fi
}
to_dataset "$W/lost_net/harvest" "$W/lost_net.npz"
to_dataset "$W/perturb/harvest" "$W/perturb.npz"
"$PY" "$S/r2d_merge.py" "$W/train2.npz" "$W1/train.npz" "$W/perturb.npz" "$W/lost_net.npz:10" || exit 4

echo "== 4. train  $(date)"
OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=16 "$PY" -u scripts/train_variant.py --data "$W/train2.npz" \
    --out "$W/ml_guess_r2d_it2.pt" --init "$NET" --lr 2e-4 --epochs 100 --batch 256 --val-frac 0.1 \
    --seed 42 --gamma 1.6666666666666667 --threads 64 --device cpu \
    --proxy-evalset "$W1/proxy.npz" --proxy-every 10 > "$W/train_it2.log" 2>&1
echo "== training exit code $?"; tail -n 3 "$W/train_it2.log"
BEST=$W/ml_guess_r2d_it2_proxy.pt
[ -f "$BEST" ] || BEST=$W/ml_guess_r2d_it2.pt

echo "== 5. paired test on held-out faces  $(date)"
cd "$HLLD_DIR" || exit 1
OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 "$PY" -u "$S/r2d_eval_seeds.py" "$W1/eval.npz" "$W/eval_it2.npz" 6000 \
    production=data/ml_guess_rotor_ft.pt it1="$NET" it2="$BEST"

echo "== 6. the ledger retried with iteration 2  $(date)"
RMHD_ML_CKPT="$BEST" "$PY" -u "$S/r2d_resolve.py" lost "$H" "$W/lost_it2" || exit 6
echo "== $(date) done"
