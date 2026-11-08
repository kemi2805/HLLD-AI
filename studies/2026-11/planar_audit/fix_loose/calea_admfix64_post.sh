#!/bin/bash
#SBATCH --job-name=admfix-post
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD/logs/slurm_%x_%j.out
# After the fixed filter reruns (admfix-r2d/ot/rotor): the same audit as the
# first round's (77970, 78316, 78317) on the new harvests -- admissibility
# flags (admiss_harvest.py), the flagged rows (extract_flagged.py), the RH
# tail (extract_rh_tail.py) -- and the face counts / wall of control, first
# filter run and fixed rerun.
set -u
B=/mnt/rafast/miler/codes/rotor2d
R=$B/HLLD/results
A=$R/r2d_retrain/planar_audit
S=/mnt/rafast/miler/scratch/r2d_retrain/planar_audit
PY=$HOME/venv/rmhd/bin/python
cd "$B/rmhd_final" || exit 1
. scripts/kernels.env
export OMP_NUM_THREADS=1 NPROC=60
RUNS="$R/r2d_64_exact_admfix $R/ot_64_exact_admfix $R/rotor_64_exact_admfix"
echo "== $(date) $(hostname)"
"$PY" -u "$S/admiss_harvest.py" "$A/admiss_harvests_admfix.json" $RUNS
"$PY" -u "$S/extract_flagged.py" "$A/flagged_admfix" $RUNS
"$PY" -u "$S/extract_rh_tail.py" "$A/rh_tail_admfix" $RUNS
"$PY" - "$R" <<'PY'
import json, os, sys
R = sys.argv[1]
for p, ctl in (("r2d", "r2d_64_exact_ctl1102"), ("ot", "ot_64_exact_ssretry"), ("rotor", "rotor_64_exact_ssretry")):
    for d in (ctl, p + "_64_exact_admiss", p + "_64_exact_admfix"):
        f = os.path.join(R, d, "run_meta.json")
        if not os.path.exists(f):
            print("%-24s no run_meta" % d); continue
        m = json.load(open(f))
        h = m.get("harvest", {}).get("faces", {})
        print("%-24s steps %4s wall %7s s  attempted %8s exact %8s lost %6s  seven %8s planar5 %7s planar4 %5s" % (
            d, m.get("steps"), m.get("wall_s"), h.get("attempted"), h.get("exact"), h.get("hlld_after_attempt"),
            h.get("seven"), h.get("planar5"), h.get("planar4")))
PY
echo "== $(date) done"
