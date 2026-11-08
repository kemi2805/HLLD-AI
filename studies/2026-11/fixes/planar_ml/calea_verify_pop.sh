#!/bin/bash
#SBATCH --job-name=verify-pop
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# verify_rmhd.py on the planar-network replay's population (2026-11-08):
# production's late-rung / four-unknown answers there (the harvest runs' rows,
# split per sweep as replay_net.py does) and the new defaults' answers (the
# netreg replay arm's harvest: network + regular answers first; the net arm's:
# network, REGULAR=0, the same code), each chunk
# with its run's gamma.  -> results/fixes/verify_pop/
#   sbatch --dependency=afterok:<netreg job>,afterok:<net job> calea_verify_pop.sh
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
OUT=$BASE/HLLD/results/fixes/verify_pop
R3=$BASE/HLLD/results/planar_ml/replay_v3
export PYTHONPATH=$BASE/rmhd_final-loose13:$S RMHD_DIR=$BASE/rmhd_final-loose13
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
. "$BASE/rmhd_final-loose13/scripts/kernels.env"
cd "$S" || exit 1
echo "== $(date) $(hostname): prep"
"$PY" -u planar_ml/verify_pop_prep.py "$OUT" --population r2d_64_exact_finalrotc7lhgate \
    ot_64_exact_finalrotc7lhgate rotor_64_exact_finalrotc7lhgate || exit 1
"$PY" -u planar_ml/verify_pop_prep.py "$OUT" --dir r2d_netreg "$R3/harvest_netreg_r2d" \
    ot0_netreg "$R3/harvest_netreg_ot_0" ot1_netreg "$R3/harvest_netreg_ot_1" \
    rotor_netreg "$R3/harvest_netreg_rotor" || exit 1
"$PY" -u planar_ml/verify_pop_prep.py "$OUT" --dir r2d_net "$R3/harvest_net_r2d" \
    ot0_net "$R3/harvest_net_ot_0" ot1_net "$R3/harvest_net_ot_1" \
    rotor_net "$R3/harvest_net_rotor" || exit 1
cp "$S/planar_ml/pop_g53.npz" "$S/planar_ml/pop_g43.npz" "$OUT/"
echo "== $(date): verify"
pids=()
for f in "$OUT"/*_g[45]3_s[23]_[0-9][0-9][0-9].npz; do
    while [ "$(jobs -rp | wc -l)" -ge 62 ]; do sleep 5; done
    case "$(basename "$f")" in *_g43_*) pop=$OUT/pop_g43.npz ;; *) pop=$OUT/pop_g53.npz ;; esac
    SP_POP=$pop "$PY" -u verify_rmhd.py "$f" "${f%.npz}_verify.npz" > "${f%.npz}_verify.log" 2>&1 &
    pids+=($!)
done
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) done, $nf verify processes failed"
"$PY" planar_ml/verify_rescues_summary.py "$OUT"
