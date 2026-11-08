#!/bin/bash
#SBATCH --job-name=verify-rmhd
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# Are our exact answers actual solutions of ideal RMHD?  (user, 2026-11-05)
#  1. every answer the production path + the loss fixes (final2, DS 1e-4) give on
#     the 4Q 128^2 failure-ledger population (2,100 faces), captured in the
#     harvest schema (capture_answers.py, 32 shards, no pool);
#  2. a uniform sample of 256,000 of the answers the 128^2 run itself produced
#     (its harvest: seven-wave + planar rows);
# each checked by verify_rmhd.py: jump conditions of every discontinuity, every
# fan re-integrated and the self-similar weak-solution identity, wave order,
# and the entropy condition -- with rmhd_ideal's fluxes, written from the
# equations (independent of the solver).
set -u
BASE=/mnt/rafast/miler/codes/rotor2d
S=/mnt/rafast/miler/scratch/fixes
PY=$HOME/venv/rmhd/bin/python
POP=$BASE/HLLD/results/failure_ledger_r2d128_ft3/population.npz
OUT=$BASE/HLLD/results/fixes/verify
RUN=$BASE/HLLD/results/r2d_128_exact_ml_r2d_it3/harvest
export HLLD_DIR=$BASE/HLLD-fix5 PYTHONPATH=$BASE/rmhd_final-loose5:$S RMHD_DIR=$BASE/rmhd_final-loose5 SP_POP=$POP
export RMHD_ML_CKPT=$BASE/HLLD/results/r2d_retrain/it3/ml_guess_r2d_it3.pt
export RMHD_ML_CKPTS=$BASE/rmhd_final/data/ml_guess_rotor_big_s42.pt,$BASE/rmhd_final/data/ml_guess_rotor_big_s47.pt
export RMHD_PLANAR5_FALLBACK=1 RETRIES=2 RMHD_POOL=0
export RMHD_SLOWSHOCK_LOOSE_RETRY=1 RMHD_PLANAR4_STAR=1 RMHD_PLANAR4_STAR_DS=1e-4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TORCH_THREADS=1
. "$BASE/rmhd_final-loose5/scripts/kernels.env"
mkdir -p "$OUT/ledger" "$OUT/run"
cd "$S" || exit 1
"$PY" -c 'import rmhd, sys; sys.exit(0 if "rmhd_final-loose5" in rmhd.__file__ else 1)' || { echo "PREFLIGHT FAILED: rmhd"; exit 2; }
echo "== $(date) $(hostname): capture"
pids=()
for ((i = 0; i < 32; i++)); do
    "$PY" -u capture_answers.py "$POP" "$OUT/ledger" --shard $i --nshards 32 > "$OUT/ledger/capture_$(printf %03d $i).log" 2>&1 &
    pids+=($!)
done
"$PY" -u sample_run_harvest.py "$RUN" "$OUT/run" 256000 32 > "$OUT/run/sample.log" 2>&1 &
pids+=($!)
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) capture + sample done, $nf failed"; cat "$OUT/run/sample.log"
"$PY" - <<'PYEOF'
import glob, numpy as np
O = "/mnt/rafast/miler/codes/rotor2d/HLLD/results/fixes/verify/ledger"
parts, src = [], []
for f in sorted(glob.glob(O + "/harvest_*/solved_*.npz")):
    z = np.load(f); parts.append({k: z[k] for k in ("U_L", "U_R", "zones", "speeds")}); src.append(np.where(z["source"] == 1, 2, 1))
for f in sorted(glob.glob(O + "/planar4_*.npz")):
    z = np.load(f); parts.append({k: z[k] for k in ("U_L", "U_R", "zones", "speeds")}); src.append(np.full(len(z["U_L"]), 3))
out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
out["source"] = np.concatenate(src)
n = len(out["U_L"])
for c, ix in enumerate(np.array_split(np.arange(n), 16)):
    np.savez(O + "/chunk_%03d.npz" % c, gamma=5.0 / 3.0, **{k: v[ix] for k, v in out.items()})
print("ledger answers:", n, "by solver (1 seven, 2 planar5, 3 planar4):", np.bincount(out["source"]).tolist())
PYEOF
echo "== $(date) verify"
pids=()
for f in "$OUT"/ledger/chunk_*.npz "$OUT"/run/chunk_*.npz; do
    "$PY" -u verify_rmhd.py "$f" "${f%.npz}_verify.npz" > "${f%.npz}_verify.log" 2>&1 &
    pids+=($!)
done
nf=0; for p in "${pids[@]}"; do wait "$p" || nf=$((nf + 1)); done
echo "== $(date) done, $nf verify processes failed"
