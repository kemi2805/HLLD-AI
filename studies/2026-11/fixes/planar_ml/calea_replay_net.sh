#!/bin/bash
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Paired replay of the held-out late-rung sweeps (planar_ml/replay_net.py):
# the three harvest runs' test block (t/t_end >= 0.8), each sweep's late-rung
# faces through the new chain (the 512^2 run's switches and kernels.env), no
# pool.  One arm per node:
#   off       production's late rung
#   net       RMHD_COPLANAR7_NET=data/ml_planar_v2a.pt, fallback all
#   netfirst  the same, fallback first (production's first branch + extras)
# Worktrees rmhd_final-loose12 (c7_patches/setup_net.sh) + HLLD-fix14.
#   sbatch --job-name=rpn-off --export=ALL,ARM=off calea_replay_net.sh
# -> results/planar_ml/replay_v2/<ARM>.npz; compare_replay.py off.npz net.npz netfirst.npz
set -u
: "${ARM:?}"
B=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes/planar_ml
OUT=$B/HLLD/results/planar_ml/replay_v2
PY=$HOME/venv/rmhd/bin/python
export RMHD=$B/rmhd_final-loose12
export PYTHONPATH=$RMHD HLLD_DIR=$B/HLLD-fix14
unset RMHD_SLOW_FAN_EXACT RMHD_LINEAR_SEED RMHD_SLOWSHOCK_ADMISSIBLE RMHD_ADMISSIBLE RMHD_POOL
unset RMHD_HARVEST_RESCUES RMHD_COPLANAR7_NET RMHD_COPLANAR7_NET_FALLBACK RMHD_COPLANAR7_CALLS RMHD_COPLANAR7_P
. "$RMHD/scripts/kernels.env"
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4 RMHD_ADMISSIBLE_GATE=1
export RMHD_PLANAR4_ROT=1 RMHD_COPLANAR7=late RMHD_PLANAR4_STAR_BATCH=1
export RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt RMHD_ML_CKPTS=data/ml_guess_rotor_big_s42.pt,data/ml_guess_rotor_big_s47.pt
case "$ARM" in
    off) ;;
    net) export RMHD_COPLANAR7_NET=data/ml_planar_v2a.pt RMHD_COPLANAR7_NET_FALLBACK=all ;;
    netfirst) export RMHD_COPLANAR7_NET=data/ml_planar_v2a.pt RMHD_COPLANAR7_NET_FALLBACK=first ;;
    *) echo "unknown ARM $ARM"; exit 2 ;;
esac
for c in $RMHD_ML_CKPT ${RMHD_ML_CKPTS//,/ } data/ml_planar_v2a.pt; do
    [ -e "$RMHD/$c" ] || { echo "PREFLIGHT FAILED: no $RMHD/$c"; exit 2; }
done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=16
mkdir -p "$OUT"
cd "$HLLD_DIR" || exit 1
echo "== $(date) $(hostname) ARM=$ARM"
"$PY" -c "import rmhd, rmhd.batched.coplanar7_b as C; print(rmhd.__file__, repr(C.NET), C.NET_FALLBACK)" \
    | tee /dev/stderr | grep -q "rmhd_final-loose12" || { echo "PREFLIGHT FAILED: rmhd is not loose12"; exit 2; }
# four parts side by side, 16 numba threads each (every arm the same layout,
# so the walls compare): the 4Q, Orszag-Tang in two halves, the rotor
RUNS="r2d_64_exact_finalrotc7lhgate ot_64_exact_finalrotc7lhgate rotor_64_exact_finalrotc7lhgate"
TAG=$ARM${MAX_SWEEPS:+_pilot$MAX_SWEEPS}
pids=()
labels=()
for spec in "r2d_64_exact_finalrotc7lhgate:" "ot_64_exact_finalrotc7lhgate:0/2" \
            "ot_64_exact_finalrotc7lhgate:1/2" "rotor_64_exact_finalrotc7lhgate:"; do
    run=${spec%%:*}
    part=${spec#*:}
    lab=${run%%_*}${part:+_${part%%/*}}
    "$PY" -u "$S/replay_net.py" "$OUT/${TAG}_$lab.npz" $RUNS --select $run ${part:+--part $part} \
        ${MAX_SWEEPS:+--max-sweeps $MAX_SWEEPS} > "$OUT/${TAG}_$lab.log" 2>&1 &
    pids+=($!)
    labels+=($lab)
done
st=0
for i in "${!pids[@]}"; do
    wait ${pids[$i]}
    s=$?
    echo "== part ${labels[$i]} exit $s"; tail -n 3 "$OUT/${TAG}_${labels[$i]}.log"
    [ $s -eq 0 ] || st=1
done
[ $st -eq 0 ] || { echo "== $(date) a part FAILED"; exit 1; }
# the parts in order -> one file per arm
"$PY" - "$OUT/$TAG.npz" "$OUT/${TAG}_r2d.npz" "$OUT/${TAG}_ot_0.npz" "$OUT/${TAG}_ot_1.npz" "$OUT/${TAG}_rotor.npz" <<'PYEOF'
import sys
import numpy as np
parts = [np.load(p, allow_pickle=True) for p in sys.argv[2:]]
out = {k: parts[0][k] for k in ("runs", "flux_keys", "env", "net", "fallback")}
for k in parts[0].files:
    if k.startswith(("face_", "sweep_")):
        out[k] = np.concatenate([p[k] for p in parts])
np.savez_compressed(sys.argv[1], **out)
print("merged %d parts: %d sweeps, %d faces -> %s" % (len(parts), out["sweep_wall"].size,
                                                      out["face_rung"].size, sys.argv[1]))
PYEOF
st=$?
echo "== $(date) exit $st"
exit $st
