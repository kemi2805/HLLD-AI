#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The planar warm-start network, first training (user 2026-11-06: "start the
# planar model training when the data is in"): planar_data.py on the four
# dataset jobs' output (81485-81488: 4Q 64^2 + 128^2, OT 64^2, rotor 64^2,
# results/planar_ml/<run>/), then planar_train.py twice side by side -- the
# classification losses at weight 1 (v1a) and 0.3 (v1b) against the
# regression's -- 32 threads each, split by time (train t/t_end < 0.7,
# validation to 0.8, test after).  -> results/planar_ml/train_v1/
#   sbatch --dependency=afterok:81485:81486:81487:81488 calea_planar_train.sh
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/planar_ml
D=$B/HLLD/results/planar_ml
OUT=$D/train_v1
PY=$HOME/venv/rmhd/bin/python
mkdir -p "$OUT"
cd "$OUT" || exit 1
echo "== $(date) $(hostname)"
"$PY" -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
"$PY" -u "$S/planar_data.py" "$OUT/data_v1.npz" $D/r2d_64_exact_ctl1102 $D/r2d_128_exact_ml_r2d_it3 \
    $D/ot_64_exact_ssretry $D/rotor_64_exact_ssretry > "$OUT/data_v1.log" 2>&1
st=$?
cat "$OUT/data_v1.log"
[ $st -eq 0 ] || { echo "== planar_data.py FAILED ($st)"; exit 1; }
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
"$PY" -u "$S/planar_train.py" "$OUT/data_v1.npz" "$OUT/planar_v1a.pt" --threads 32 --cls-weight 1.0 > "$OUT/train_v1a.log" 2>&1 &
pa=$!
"$PY" -u "$S/planar_train.py" "$OUT/data_v1.npz" "$OUT/planar_v1b.pt" --threads 32 --cls-weight 0.3 > "$OUT/train_v1b.log" 2>&1 &
pb=$!
wait $pa; sa=$?
wait $pb; sb=$?
for v in a b; do echo "######## v1$v"; grep -v -i warn "$OUT/train_v1$v.log" | tail -n 14; done
echo "== $(date) exit v1a $sa v1b $sb"
[ $sa -eq 0 ] && [ $sb -eq 0 ]
