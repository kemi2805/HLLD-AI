#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The planar warm-start network, second training (user 2026-11-06: "start the
# planar model training when the data is in").
#   data     v1's four planar dataset jobs + the three harvest runs (81497-81499:
#            4Q, OT, rotor 64^2, late chain with RMHD_HARVEST_RESCUES=1; their
#            five-wave, late-rung and four-unknown rows), every row with its
#            solver-frame states (planar_data.py)
#   baseline production's seed (ml_guess_rotor_ft.pt, as the late rung gets it)
#            on the validation and test rows, in the targets' units
#            (planar_baseline.py, 32 workers)
#   train    v2a 256-256-128 and v2b 512-512-256 side by side, 32 threads each:
#            early stop on the REGRESSION loss (v1 stopped on the sum, which
#            the rising rotation loss ended early), classification weight 0.3,
#            300 epochs, patience 40; the report puts the baseline's errors next
#            to the network's on the same rows
# -> results/planar_ml/train_v2/
#   sbatch --dependency=afterok:81498:81499 calea_planar_train2.sh
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/planar_ml
D=$B/HLLD/results/planar_ml
OUT=$D/train_v2
PY=$HOME/venv/rmhd/bin/python
export HLLD_DIR=$B/HLLD-fix14 PYTHONPATH=$B/rmhd_final-loose11
unset RMHD_PLANAR_NJIT RMHD_MERGE_SWEEPS
mkdir -p "$OUT"
cd "$OUT" || exit 1
echo "== $(date) $(hostname)"
"$PY" -c "import numpy, torch, rmhd; print('numpy', numpy.__version__, 'torch', torch.__version__, 'rmhd', rmhd.__file__)"
"$PY" -c "import rmhd.batched.coplanar7_b" || { echo "PREFLIGHT FAILED: rmhd is not the loose11 worktree"; exit 2; }
"$PY" -u "$S/planar_data.py" "$OUT/data_v2.npz" $D/r2d_64_exact_ctl1102 $D/r2d_128_exact_ml_r2d_it3 \
    $D/ot_64_exact_ssretry $D/rotor_64_exact_ssretry harvest:r2d_64_exact_finalrotc7lhgate \
    harvest:ot_64_exact_finalrotc7lhgate harvest:rotor_64_exact_finalrotc7lhgate > "$OUT/data_v2.log" 2>&1
st=$?
cat "$OUT/data_v2.log"
[ $st -eq 0 ] || { echo "== planar_data.py FAILED ($st)"; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -u "$S/planar_baseline.py" "$OUT/data_v2.npz" "$OUT/baseline_v2.npz" --workers 32 \
    > "$OUT/baseline_v2.log" 2>&1
st=$?
grep -v -i warn "$OUT/baseline_v2.log" | tail -n 4
[ $st -eq 0 ] || { echo "== planar_baseline.py FAILED ($st)"; exit 1; }
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
common="--threads 32 --cls-weight 0.3 --stop-on reg --epochs 300 --patience 40 --baseline $OUT/baseline_v2.npz"
"$PY" -u "$S/planar_train.py" "$OUT/data_v2.npz" "$OUT/planar_v2a.pt" $common --hidden 256,256,128 \
    > "$OUT/train_v2a.log" 2>&1 &
pa=$!
"$PY" -u "$S/planar_train.py" "$OUT/data_v2.npz" "$OUT/planar_v2b.pt" $common --hidden 512,512,256 \
    > "$OUT/train_v2b.log" 2>&1 &
pb=$!
wait $pa; sa=$?
wait $pb; sb=$?
for v in a b; do echo "######## v2$v"; grep -v -i warn "$OUT/train_v2$v.log" | tail -n 14; done
echo "== $(date) exit v2a $sa v2b $sb"
[ $sa -eq 0 ] && [ $sb -eq 0 ]
