#!/bin/bash
#SBATCH --job-name=tubes-examples
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Lost-face examples (user 2026-11-04: "visual examples of the loss"): the 1D
# tube of each chosen ledger lane at 1024, 2048 and 4096 cells, one process each.
set -u
S=/mnt/rafast/miler/scratch/r2d_retrain/anatomy
LED=/mnt/rafast/miler/codes/rotor2d/HLLD/results/failure_ledger_r2d128_ft3
OUT=/mnt/rafast/miler/codes/rotor2d/HLLD/results/r2d_retrain/anatomy128/tubes
PY=$HOME/venv/rmhd/bin/python
export HLLD_DIR=/mnt/rafast/miler/codes/rotor2d/HLLD-harvest
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_THREADS=2
mkdir -p "$OUT"
echo "== $(date) $(hostname)  lanes: ${LANES:?}"
pids=()
for L in $LANES; do
    for N in 1024 2048 4096; do
        "$PY" -u "$S/tubes_examples.py" "$LED" "$L" "$N" "$OUT/tube_${L}_${N}.npz" > "$OUT/tube_${L}_${N}.log" 2>&1 &
        pids+=($!)
    done
done
for p in "${pids[@]}"; do wait "$p"; done
cat "$OUT"/tube_*.log
echo "== $(date) done"
