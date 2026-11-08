#!/bin/bash
#SBATCH --job-name=adm64-post
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# After the paired 64^2 runs (adm64-r2d/ot/rotor): the admissibility audit of
# the treatment harvests (admiss_harvest.py; the controls' numbers are in
# admiss_harvests_1103.json), and both runs' face counts and wall clock.
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
PY=$HOME/venv/rmhd/bin/python
cd "$B/rmhd_final" || exit 1
. scripts/kernels.env
export OMP_NUM_THREADS=1 NPROC=60
echo "== $(date) $(hostname)"
"$PY" -u /mnt/rafast/miler/scratch/r2d_retrain/planar_audit/admiss_harvest.py \
    "$R/r2d_retrain/planar_audit/admiss_harvests_adm64.json" \
    "$R/r2d_64_exact_admiss" "$R/ot_64_exact_admiss" "$R/rotor_64_exact_admiss"
"$PY" - "$R" <<'PY'
import json, os, sys
R = sys.argv[1]
for ctl, adm in (("r2d_64_exact_ctl1102", "r2d_64_exact_admiss"), ("ot_64_exact_ssretry", "ot_64_exact_admiss"),
                 ("rotor_64_exact_ssretry", "rotor_64_exact_admiss")):
    for d in (ctl, adm):
        p = os.path.join(R, d, "run_meta.json")
        if not os.path.exists(p):
            print("%-24s no run_meta" % d); continue
        m = json.load(open(p))
        f = m.get("harvest", {}).get("faces", {})
        print("%-24s steps %4s wall %7s s  attempted %8s exact %8s lost %6s  seven %8s planar5 %7s planar4 %5s" % (
            d, m.get("steps"), m.get("wall_s"), f.get("attempted"), f.get("exact"), f.get("hlld_after_attempt"),
            f.get("seven"), f.get("planar5"), f.get("planar4")))
PY
echo "== $(date) done"
