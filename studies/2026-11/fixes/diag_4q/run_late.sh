#!/bin/bash
# local: the planar seven-wave stage as a rung after the five-wave solver (RMHD_COPLANAR7=late)
#  1. the 96 early-4Q faces the control answered with the five-wave solver (production network): late vs off (19 s) / after (79 s)
#  2. the 24-face ledger smoke (8 refused + 3 still-lost + 13 controls, it3 network): late and after vs first (smoke_c7_1)
set -u
F=/Users/miler/Codes/claude_scratch/r2d_retrain/fixes
. /Users/miler/Codes/rmhd_final/scripts/kernels.env
export HLLD_DIR=$F/hlld_fix PYTHONPATH=$F/rmhd_fix
export RMHD_ML_CKPTS=/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s42.pt,/Users/miler/Codes/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2 OMP_NUM_THREADS=1 RMHD_POOL=0
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1 RMHD_PLANAR4_ROT=1
run() {   # run <pop> <out> <ckpt> <mode>
    t0=$(date +%s)
    RMHD_ML_CKPT=$3 RMHD_COPLANAR7=$4 /Users/miler/Codes/py-work/bin/python -u $F/replay_pop.py $1 $2 --chunk 32 2>&1 | grep -v -i warn | tail -n 1
    echo "  $(basename $2) mode $4: $(( $(date +%s) - t0 )) s"
}
run $F/diag_4q/pop_p5.npz $F/diag_4q/p5_late.npz /Users/miler/Codes/rmhd_final/data/ml_guess_rotor_ft.pt late
IT3=/Users/miler/Codes/claude_scratch/r2d_retrain/smoke_audit/it3/ml_guess_r2d_it3.pt
run $F/smoke_c7_pop.npz $F/diag_4q/smoke_late.npz $IT3 late
run $F/smoke_c7_pop.npz $F/diag_4q/smoke_after.npz $IT3 after
