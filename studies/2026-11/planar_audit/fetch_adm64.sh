#!/bin/bash
# fetch_adm64.sh: the paired 64^2 runs (control, treatment) of the three
# problems into runs_adm64/<problem>_<ctl|adm>/ for adm_compare64.py, plus the
# 512^2 HLLD references (snap_fin) into runs_adm64/ref_<problem>.npz.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=/mnt/rafast/miler/codes/rotor2d/HLLD
RUNS="r2d_ctl:r2d_64_exact_ctl1102:slurm_r2d-ctl64_75443.out
r2d_adm:r2d_64_exact_admiss:slurm_adm64-r2d_77965.out
ot_ctl:ot_64_exact_ssretry:slurm_orsza-ssr_60939.out
ot_adm:ot_64_exact_admiss:slurm_adm64-ot_77966.out
rotor_ctl:rotor_64_exact_ssretry:slurm_rotor-ssr_60938.out
rotor_adm:rotor_64_exact_admiss:slurm_adm64-rotor_77967.out"
[ $# -gt 0 ] && RUNS="$*"
for line in $RUNS; do
    IFS=: read -r name dir log <<< "$line"
    dst=$HERE/runs_adm64/$name
    mkdir -p "$dst/harvest"
    rsync -a --include='run_meta.json' --include='diag.csv' --include='snap_*.npz' \
        --exclude='*' "itp:$R/results/$dir/" "$dst/" < /dev/null || echo "rsync $dir failed"
    rsync -a --include='coverage_*.npz' --exclude='*' \
        "itp:$R/results/$dir/harvest/" "$dst/harvest/" < /dev/null || echo "rsync $dir/harvest failed"
    rsync -a "itp:$R/logs/$log" "$dst/slurm.out" < /dev/null || echo "rsync $log failed"
    echo "$name <- $dir: $(ls "$dst"/harvest | wc -l | tr -d ' ') coverage shards, $(ls "$dst"/snap_*.npz 2>/dev/null | wc -l | tr -d ' ') snapshots"
done
for p in r2d ot rotor; do
    ref=${p}_512_hlld; [ $p = ot ] && ref=ot_512_hlld_c2pfix
    [ -f "$HERE/runs_adm64/ref_$p.npz" ] || rsync -a "itp:$R/results/$ref/snap_fin.npz" \
        "$HERE/runs_adm64/ref_$p.npz" < /dev/null
done
