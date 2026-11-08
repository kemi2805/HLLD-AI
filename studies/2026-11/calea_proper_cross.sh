#!/bin/bash
#SBATCH --job-name=ml-proper-cross
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# After the proper retrains of the rotor and the Orszag-Tang
# (calea_proper_2d.sh): every chosen network on every problem's TEST faces --
# the rotor's, the Orszag-Tang's and the four-quadrant's (job 77245's proper
# split) -- with production's network, one attempt, the same faces as each
# problem's own test (r2d_eval_seeds.py draws them with a fixed seed).  Does
# one network serve all three problems, or does each need its own?
#   sbatch --dependency=afterok:<rotor job>:<ot job> calea_proper_cross.sh
set -u
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/r2d_retrain
R=$B/HLLD/results
Q=$R/r2d_retrain
M=$R/ml_retrain
HL=$B/HLLD
WT=$B/rmhd_final-valsplit
PY=$HOME/venv/rmhd/bin/python
INIT=$B/rmhd_final/data/ml_guess_rotor_ft.pt
cd "$WT" || exit 1
export PYTHONPATH=$WT
unset RMHD_ML_CKPT RMHD_ML_CKPTS RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED
. scripts/kernels.env
read -r RN RP < "$M/rotor/chosen.txt" || exit 2
read -r ON OP < "$M/ot/chosen.txt" || exit 2
echo "== $(date) $(hostname)  rotor: $RN = $RP;  ot: $ON = $OP"
ARMS=(production="$INIT" q4_proper="$Q/proper/ml_guess_r2d_p_run_pert.pt" rotor_net="$RP" ot_net="$OP")
PAIRS=(rotor_net:production ot_net:production q4_proper:production rotor_net:ot_net rotor_net:q4_proper ot_net:q4_proper)
# Added after the rotor's test (post hoc, not a choice): VAL chose the rotor's
# p_run_proxy, while on TEST p_pert was ahead in both time blocks -- so each
# problem's p_pert rides along wherever VAL chose another checkpoint
for P in rotor ot; do
    PP=$M/$P/ml_guess_${P}_p_run_pert.pt
    if [ "$P" = rotor ]; then C=$RP; else C=$OP; fi
    if [ "$C" != "$PP" ]; then
        ARMS+=(${P}_pert="$PP")
        PAIRS+=(${P}_pert:production ${P}_pert:${P}_net)
    fi
done
echo "   arms: ${ARMS[*]}"
pids=()
for P in rotor ot q4; do
    if [ "$P" = q4 ]; then T=$Q/proper/test.npz; else T=$M/$P/test.npz; fi
    HLLD_DIR=$HL OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=20 "$PY" -u "$S/r2d_eval_seeds.py" \
        "$T" "$M/cross_$P.npz" 6000 "${ARMS[@]}" > "$M/cross_$P.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do
    wait "$p" || { echo "an evaluation failed (pid $p)"; tail -n 5 "$M"/cross_*.log; exit 3; }
done
for P in rotor ot q4; do
    if [ "$P" = q4 ]; then T=$Q/proper/test.npz; X=$Q/proper/proxy_val.npz
    else T=$M/$P/test.npz; X=$M/$P/proxy_val.npz; fi
    echo "== $P TEST"
    cat "$M/cross_$P.log"
    "$PY" "$S/r2d_paired.py" "$M/cross_$P.npz" "$T" "$X" "${PAIRS[@]}" || exit 4
done
echo "== $(date) done"
