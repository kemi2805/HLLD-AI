#!/bin/bash
#SBATCH --job-name=wide-window
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Would intermediate shocks let the seven-wave solver answer the faces that
# need the planar solver?  (user 2026-11-04)  With the slow-shock window
# opened past the Alfven speed (MARGIN 1.0, numpy path) and, for comparison,
# production's window (0.03, numpy path):
#   1. conditioning.py: the seven-wave Jacobian at the known answers
#      (first / retry / planar faces of six solved shards, 2000 each)
#   2. seed_ceiling_fine.py: the seven-wave Newton from the known answer
#      moved by 0, 3e-4, 1e-3, 3e-3 (1500 per group, the 78408 draw)
set -u
S=/mnt/rafast/miler/scratch/r2d_retrain/anatomy
H=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_128_exact_ml_r2d_it3/harvest
OUT=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/anatomy128/wide
HD=/mnt/rafast/miler/codes/rotor2d/HLLD-harvest
PY=$HOME/venv/rmhd/bin/python
unset RMHD_SLOWSHOCK RMHD_FAN RMHD_ALFVEN RMHD_SHOCK
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4 TORCH_THREADS=4
export HLLD_DIR=$HD PYTHONPATH=$HD
mkdir -p "$OUT"
cd "$S" || exit 1
SH="$H/solved_r000_00030.npz $H/solved_r000_00080.npz $H/solved_r000_00130.npz $H/solved_r000_00180.npz $H/solved_r000_00220.npz $H/solved_r000_00250.npz"
echo "== $(date) $(hostname)"
for M in 1.0 0.03; do
    WIDE_MARGIN=$M "$PY" -u run_wide.py conditioning.py $SH --n 2000 --out "$OUT/cond_margin$M.npz" \
        > "$OUT/cond_margin$M.log" 2>&1 &
    ( cd "$HD" && WIDE_MARGIN=$M "$PY" -u "$S/run_wide.py" "$S/seed_ceiling_fine.py" "$H" --n 1500 \
        --radius 0 0.0003 0.001 0.003 --draws 1 --shrink 0.1 \
        --ckpts /mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt \
        --out "$OUT/seed_margin$M.npz" > "$OUT/seed_margin$M.log" 2>&1 ) &
done
wait
for f in "$OUT"/*.log; do echo "== $f"; grep -vE "Warning|W = xp|warn" "$f" | tail -n 14; done
echo "== $(date) done"
