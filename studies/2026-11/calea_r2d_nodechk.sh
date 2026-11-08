#!/bin/bash
#SBATCH --job-name=r2d-nodechk
#SBATCH --partition=calea
#SBATCH --nodelist=calea01
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Node-speed check for the 64^2 wall-time comparison.  The new-network run
# (75450) ran on calea01; the control (75443) and the 10-29 run (60940) ran on
# calea03 and agree there to 0.2% (step 40) / 0.6% (step 100).  This replays
# the control -- production's network, the same code, TEND 0.4 and NSNAP 8 --
# on calea01 for its first 100 steps: dt does not depend on the node, so the
# work is the control's exactly (snap_001 at step 46 and snap_002 at step 92
# compare bitwise), and the wall clock per step against the control's log is
# the speed of calea01 relative to calea03.
set -u
B=/mnt/rafast/miler/codes/rotor2d
unset PYTHONPATH HLLD_DIR RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_ML_CKPT RMHD_ML_CKPTS
export PROBLEM=riemann2d N=64 MAX_STEPS=100 TAG=_nodechk01
cd "$B/HLLD" || exit 1
exec bash scripts/calea_rotor_exact.sh
