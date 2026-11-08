#!/bin/bash
#SBATCH --job-name=tubes-hi
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# 1D tubes (HLLD + MC, 8192 and 16384 cells, t = 0.18) of lanes whose answers carry field-reversing slow jumps --
# entropy falling (214 788 1013 1245 1299) and rising (1148 1406): does the tube keep converging to the exact flux?
set -u
S=/mnt/rafast/miler/scratch/r2d_retrain/anatomy
LED=/mnt/rafast/miler/codes/rotor2d/HLLD/results/failure_ledger_r2d128_ft3
OUT=/mnt/rafast/miler/codes/rotor2d/HLLD/results/fixes/tubes_hi
PY=$HOME/venv/rmhd/bin/python
export HLLD_DIR=/mnt/rafast/miler/codes/rotor2d/HLLD-harvest
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_THREADS=2
mkdir -p "$OUT"
echo "== $(date) $(hostname)  lanes: ${LANES:?}"
pids=()
for L in $LANES; do
    for N in 8192 16384; do
        "$PY" -u "$S/tubes_examples.py" "$LED" "$L" "$N" "$OUT/tube_${L}_${N}.npz" > "$OUT/tube_${L}_${N}.log" 2>&1 &
        pids+=($!)
    done
done
for p in "${pids[@]}"; do wait "$p"; done
cat "$OUT"/tube_*.log
echo "== $(date) done"
