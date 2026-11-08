#!/bin/bash
# fetch_runs64.sh: copy what r2d_compare64.py reads of the four-quadrant 64^2
# exact runs from calea (run_meta.json, diag.csv, snapshots, the coverage
# bitmaps -- not the solved/unsolved rows -- and the Slurm log with the
# per-step wall clock) into runs64/<name>/.  Safe to rerun on a running job.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
R=/mnt/rafast/miler/codes/rotor2d/HLLD
# name  results dir  slurm log; or as arguments, e.g.
#   fetch_runs64.sh ft3_128:r2d_128_exact_ml_r2d_it3:slurm_r2d-ft3_77300.out
RUNS="oct29:r2d_64_exact_ssretry:slurm_riema-ssr_60940.out
ctl:r2d_64_exact_ctl1102:slurm_r2d-ctl64_75443.out
it3:r2d_64_exact_ml_r2d_it3:slurm_r2d-ml64_75450.out"
[ $# -gt 0 ] && RUNS="$*"
for line in $RUNS; do
    IFS=: read -r name dir log <<< "$line"
    dst=$HERE/runs64/$name
    mkdir -p "$dst/harvest"
    rsync -a --include='run_meta.json' --include='diag.csv' --include='snap_*.npz' \
        --exclude='*' "itp:$R/results/$dir/" "$dst/" < /dev/null || echo "rsync $dir failed"
    rsync -a --include='coverage_*.npz' --exclude='*' \
        "itp:$R/results/$dir/harvest/" "$dst/harvest/" < /dev/null || echo "rsync $dir/harvest failed"
    rsync -a "itp:$R/logs/$log" "$dst/slurm.out" < /dev/null || echo "rsync $log failed"
    echo "$name <- $dir: $(ls "$dst"/harvest | wc -l | tr -d ' ') coverage shards, $(ls "$dst"/snap_*.npz 2>/dev/null | wc -l | tr -d ' ') snapshots"
done
