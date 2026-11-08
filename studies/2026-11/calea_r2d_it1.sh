#!/bin/bash
#SBATCH --job-name=r2d-retrain1
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-08:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Four-quadrant retrain loop, iteration 1 (user, 2026-11-02: "immediately go
# with the retrain ... four quadrant only ... retrain on calea"):
#   1. the solved faces of results/r2d_64_exact_ssretry (today's solver) as a
#      dataset, with the recovered time of every row;
#   2. r2d_split.py: t = 0.175 / 0.325 +- 0.0125 held out (eval.npz, proxy.npz),
#      the rest for training with the complex faces (rotation > 0.5 rad) x10;
#   3. two networks side by side, each picked by the seven-wave Newton on the
#      proxy faces (train_variant.py --proxy-evalset): ft1 = production's
#      rotor_ft fine-tuned, scratch1 = the same size from scratch;
#   4. r2d_eval_seeds.py: the paired one-attempt test on held-out faces
#      against production's network and the neutral guess.
# Code: rmhd_final-slowfan (= main 627c7e8), HLLD-linseed (= main 8fc5b12),
# every RMHD switch at its default.
#   ssh itp; sbatch /mnt/rafast/miler/scratch/r2d_retrain/calea_r2d_it1.sh
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
W=$R/r2d_retrain/it1
S=/mnt/rafast/miler/scratch/r2d_retrain
RF=$B/rmhd_final-slowfan
PY=$HOME/venv/rmhd/bin/python
export PYTHONPATH=$RF HLLD_DIR=$B/HLLD-linseed
unset RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED
mkdir -p "$W"
cd "$RF" || exit 1
. scripts/kernels.env
echo "== $(date) $(hostname)  rmhd_final $(git rev-parse --short HEAD) + [$(git status --short | tr '\n' ' ')]"

echo "== 1. dataset  $(date)"
OMP_NUM_THREADS=8 "$PY" scripts/harvest_to_dataset.py "$R/r2d_64_exact_ssretry/harvest" \
    --out "$W/all.npz" --keep-tags --hold-out-t 99 || exit 2
echo "== 2. split  $(date)"
"$PY" "$S/r2d_split.py" "$W/all.npz" "$W" --dup 10 --rot 0.5 || exit 3

echo "== 3. train  $(date)"
COMMON=(--data "$W/train.npz" --batch 256 --val-frac 0.1 --seed 42 --gamma 1.6666666666666667
        --threads 32 --device cpu --proxy-evalset "$W/proxy.npz")
OMP_NUM_THREADS=32 NUMBA_NUM_THREADS=16 "$PY" -u scripts/train_variant.py "${COMMON[@]}" \
    --out "$W/ml_guess_r2d_ft1.pt" --init data/ml_guess_rotor_ft.pt --lr 2e-4 --epochs 150 \
    --proxy-every 10 > "$W/train_ft1.log" 2>&1 &
p1=$!
OMP_NUM_THREADS=32 NUMBA_NUM_THREADS=16 "$PY" -u scripts/train_variant.py "${COMMON[@]}" \
    --out "$W/ml_guess_r2d_scratch1.pt" --hidden 256 256 128 --lr 1e-3 --epochs 300 \
    --proxy-every 20 > "$W/train_scratch1.log" 2>&1 &
p2=$!
wait $p1; rc1=$?
wait $p2; rc2=$?
echo "== training exit codes: ft1 $rc1  scratch1 $rc2  $(date)"
tail -n 3 "$W/train_ft1.log" "$W/train_scratch1.log"

echo "== 4. paired test on held-out faces  $(date)"
ARMS=(production=data/ml_guess_rotor_ft.pt)
for v in ft1 scratch1; do
    for f in "$W/ml_guess_r2d_${v}.pt" "$W/ml_guess_r2d_${v}_proxy.pt"; do
        [ -f "$f" ] && ARMS+=("$(basename "$f" .pt | sed 's/ml_guess_r2d_//')=$f")
    done
done
cd "$HLLD_DIR" || exit 1
OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 "$PY" -u "$S/r2d_eval_seeds.py" "$W/eval.npz" "$W/eval_it1.npz" 6000 "${ARMS[@]}"
echo "== $(date) done"
