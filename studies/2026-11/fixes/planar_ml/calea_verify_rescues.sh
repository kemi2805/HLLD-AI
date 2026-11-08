#!/bin/bash
#SBATCH --job-name=verify-rescues
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# verify_rmhd.py on every late-rung and four-unknown answer of the three
# harvest runs (finalrotc7lhgate, the late chain), one process per chunk.
# Why (11-08): in the planar-network replay, one OT face's late-rung answer
# failed the verifier with and without the network (left slow and right fast
# fans), and one rotor face's four-unknown answer from the run failed it too.
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
OUT=$BASE/HLLD/results/fixes/verify_rescues
export PYTHONPATH=$BASE/rmhd_final-loose12:$S RMHD_DIR=$BASE/rmhd_final-loose12
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
. "$BASE/rmhd_final-loose12/scripts/kernels.env"
cd "$S" || exit 1
echo "== $(date) $(hostname): prep"
"$PY" -u planar_ml/verify_rescues_prep.py "$OUT" r2d_64_exact_finalrotc7lhgate ot_64_exact_finalrotc7lhgate \
    rotor_64_exact_finalrotc7lhgate || exit 1
echo "== $(date): verify"
# sol_profiles.py takes GAMMA from $SP_POP at import, and verify_rmhd.py
# re-integrates the fans with it: each chunk needs its run's gamma (OT 4/3),
# or every OT fan fails spuriously (found 11-08 on OT face 13439)
cp "$S/planar_ml/pop_g53.npz" "$S/planar_ml/pop_g43.npz" "$OUT/"
pids=()
for f in "$OUT"/*_s[23]_[0-9][0-9][0-9].npz; do
    while [ "$(jobs -rp | wc -l)" -ge 62 ]; do sleep 5; done
    case "$(basename "$f")" in ot_*) pop=$OUT/pop_g43.npz ;; *) pop=$OUT/pop_g53.npz ;; esac
    SP_POP=$pop "$PY" -u verify_rmhd.py "$f" "${f%.npz}_verify.npz" > "${f%.npz}_verify.log" 2>&1 &
    pids+=($!)
done
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) done, $nf verify processes failed"
"$PY" planar_ml/verify_rescues_summary.py "$OUT"
