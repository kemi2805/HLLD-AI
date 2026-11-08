#!/bin/bash
#SBATCH --job-name=r2d-retrain3
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-12:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Four-quadrant retrain loop, iteration 3 (user, 2026-11-02: "after you have
# ft2 retrain again with the newly won faces").  Runs after iteration 2
# (sbatch --dependency=afterok:<it2>).
#   1. the newly won faces: ledger faces iteration 2's network solved
#      (it2/lost_it2) that iteration 1's had not (it2/lost_net, the ledger
#      rows train2 already holds)
#   2. train3 = train2 + the newly won rows x10
#   3. iteration 2's network fine-tuned on train3; the final epoch is the
#      clean checkpoint (the proxy set overlaps the held-out test faces), the
#      proxy pick is tested beside it.  No newly won faces: no ft3, steps 2,
#      3 and 6 are skipped
#   4. the paired test on iteration 1's held-out faces
#   5. the network for the 64^2 run (r2d_choose.py -> chosen.txt): the newest
#      of ft3 / ft2 / ft1 unless an older one is significantly better on the
#      natural mix
#   6. the ledger retried with ft3
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
W1=$R/r2d_retrain/it1
W2=$R/r2d_retrain/it2
W=$R/r2d_retrain/it3
S=/mnt/rafast/miler/scratch/r2d_retrain
RF=$B/rmhd_final-slowfan
PY=$HOME/venv/rmhd/bin/python
H=$R/r2d_64_exact_ssretry/harvest
export PYTHONPATH=$RF HLLD_DIR=$B/HLLD-linseed
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED
export RMHD_ML_CKPTS=data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt
export RMHD_POOL=63 RMHD_POOL_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
mkdir -p "$W"
rm -f "$W/chosen.txt"
. "$RF/scripts/kernels.env"
FT1=$W1/ml_guess_r2d_ft1.pt
FT2=$W2/ml_guess_r2d_it2.pt
FT2P=$W2/ml_guess_r2d_it2_proxy.pt
INIT=$FT2P                                   # the network iteration 2 used (its BEST)
[ -f "$INIT" ] || INIT=$FT2
FT3=$W/ml_guess_r2d_it3.pt
FT3P=$W/ml_guess_r2d_it3_proxy.pt
echo "== $(date) $(hostname)  init=$INIT"

echo "== 1. newly won faces  $(date)"
cd "$RF" || exit 1
NEW=0
if compgen -G "$W2/lost_it2/harvest/solved_*.npz" > /dev/null; then
    "$PY" scripts/harvest_to_dataset.py "$W2/lost_it2/harvest" --out "$W/lost_it2.npz" || exit 1
    "$PY" "$S/r2d_newly_won.py" "$W/newly_won.npz" "$W/lost_it2.npz" "$W2/lost_net.npz" || exit 1
    NEW=$("$PY" -c "import numpy as np; print(np.load('$W/newly_won.npz')['U_L'].shape[0])") || exit 1
else
    echo "  $W2/lost_it2/harvest: no solved rows"
fi
echo "== newly won rows: $NEW"

if [ "$NEW" -gt 0 ]; then
    echo "== 2. dataset  $(date)"
    "$PY" "$S/r2d_merge.py" "$W/train3.npz" "$W2/train2.npz" "$W/newly_won.npz:10" || exit 2

    echo "== 3. train  $(date)"
    OMP_NUM_THREADS=64 NUMBA_NUM_THREADS=16 "$PY" -u scripts/train_variant.py --data "$W/train3.npz" \
        --out "$FT3" --init "$INIT" --lr 2e-4 --epochs 100 --batch 256 --val-frac 0.1 \
        --seed 42 --gamma 1.6666666666666667 --threads 64 --device cpu \
        --proxy-evalset "$W1/proxy.npz" --proxy-every 10 > "$W/train_it3.log" 2>&1
    echo "== training exit code $?"; tail -n 3 "$W/train_it3.log"
    [ -f "$FT3" ] || exit 3
else
    echo "== no newly won faces: nothing to retrain on, no ft3 (steps 2, 3, 6 skipped)"
fi

echo "== 4. paired test on held-out faces  $(date)"
cd "$HLLD_DIR" || exit 1
ARMS=(production=data/ml_guess_rotor_ft.pt ft1="$FT1" ft2="$FT2")
[ -f "$FT2P" ] && ARMS+=(ft2p="$FT2P")
[ -f "$FT3" ] && ARMS+=(ft3="$FT3")
[ -f "$FT3P" ] && ARMS+=(ft3p="$FT3P")
OMP_NUM_THREADS=16 NUMBA_NUM_THREADS=16 "$PY" -u "$S/r2d_eval_seeds.py" "$W1/eval.npz" "$W/eval_it3.npz" 6000 "${ARMS[@]}" || exit 4
if [ -f "$FT3" ]; then
    PAIRS=(ft3:ft2 ft3:ft1 ft2:ft1 ft3:production ft3:neutral)
    CANDS=(ft3="$FT3" ft2="$FT2" ft1="$FT1")
else
    PAIRS=(ft2:ft1 ft2:production ft2:neutral)
    CANDS=(ft2="$FT2" ft1="$FT1")
fi
"$PY" "$S/r2d_paired.py" "$W/eval_it3.npz" "$W1/eval.npz" "$W1/proxy.npz" "${PAIRS[@]}" 2>&1 | grep -v "no proxy"

echo "== 5. the network for the 64^2 run  $(date)"
"$PY" "$S/r2d_choose.py" "$W/eval_it3.npz" "$W/chosen.txt" "${CANDS[@]}" || exit 5

if [ -f "$FT3" ]; then
    echo "== 6. the ledger retried with ft3  $(date)"
    RMHD_ML_CKPT="$FT3" "$PY" -u "$S/r2d_resolve.py" lost "$H" "$W/lost_it3" || exit 6
fi
echo "== $(date) done"
