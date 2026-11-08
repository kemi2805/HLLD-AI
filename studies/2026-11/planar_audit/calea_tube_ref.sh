#!/bin/bash
#SBATCH --job-name=r2d-tuberef
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --mem=0
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# 1D tube reference for the flagged faces the refusal (B) left to HLLD:
# is the inadmissible exact flux (A) or HLLD (B) closer to the evolved
# solution?  Plus 200 calibration faces with a known admissible exact flux.
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
O=$B/HLLD/results/r2d_retrain/planar_audit/replay
PY=$HOME/venv/rmhd/bin/python
export PYTHONPATH=$B/rmhd_final-admiss HLLD_DIR=$B/HLLD-admiss TORCH_THREADS=1 OMP_NUM_THREADS=1 NPROC=62
cd "$HLLD_DIR" || exit 1
echo "== $(date) $(hostname)"
"$PY" -u "$S/tube_ref.py" "$O" A_report B_refuse "$O/tube_B_refuse.npz" --ncells 256,512 --chunk 4 || exit 2
echo "== $(date) done"
