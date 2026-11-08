#!/bin/bash
#SBATCH --job-name=drop-inadm
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# The training files of the proper retrains without inadmissible or no-root answers
# (user 2026-11-04: "drop them from the training set only, if they are still
# better than HLLD we keep them in the run"): drop_inadmissible.py on
# train_run.npz and train_run_pert.npz of the four-quadrant (77245), rotor
# (78178) and Orszag-Tang (78179) splits -> *_adm.npz + *_adm.json beside
# them.  VAL / TEST / proxy_val are not touched.
set -u
S=/mnt/rafast/miler/scratch/r2d_retrain
R=/mnt/rafast/miler/codes/rotor2d/HLLD/results
PY=$HOME/venv/rmhd/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "== $(date) $(hostname)"
pids=()
for D in $R/r2d_retrain/proper $R/ml_retrain/rotor $R/ml_retrain/ot; do
    for F in train_run train_run_pert; do
        [ -f "$D/$F.npz" ] || { echo "missing $D/$F.npz"; exit 2; }
        "$PY" -u "$S/drop_inadmissible.py" "$D/$F.npz" "$D/${F}_adm.npz" "$D/${F}_adm.json" \
            > "$D/${F}_adm.log" 2>&1 &
        pids+=($!)
    done
done
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "a drop failed (pid $p)"; fail=1; }; done
for D in $R/r2d_retrain/proper $R/ml_retrain/rotor $R/ml_retrain/ot; do
    for F in train_run train_run_pert; do echo "-- $D/$F"; cat "$D/${F}_adm.log"; done
done
echo "== $(date) done (fail=$fail)"
exit $fail
