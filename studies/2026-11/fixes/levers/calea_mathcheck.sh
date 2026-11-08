#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Why are the compiled planar kernels bitwise on the Mac but not on calea
# (timing arms 84463/84466, 2026-11-07)?  numpy vs numba elementary math here.
"$HOME/venv/rmhd/bin/python" -u /mnt/rafast/miler/scratch/fixes/levers/mathcheck.py
