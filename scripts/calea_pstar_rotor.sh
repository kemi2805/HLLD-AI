#!/bin/bash
#SBATCH --job-name=pstar-rotor
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD-pstar/logs/slurm_%x_%j.out
# The magnetic rotor with the p* network (run_2d.py --solver hlld_ai), on
# ONE calea node, in stages:
#
#   proof    main's 2D hlld path is unchanged: a short 64^2 rotor
#            (t = 0.05, every step logged, 4 snapshots) with `git archive
#            origin/main` of THIS clone and with this branch, plain and with
#            --face-stats; diag.csv and every snapshot array compared as raw
#            bytes.  Then rmhd_final's gate.sh bitwise (exact flux) with
#            H = this clone.  Any difference stops the job here.
#   runs     to t = 0.4, the production rotor set-up (PLM/MC, SSP-RK3, CFL
#            0.25, upwind CT, 8 snapshots -- run_2d.py's defaults, as in
#            results/rotor_{64,256}_hlld_mc), side by side:
#              hlld          64^2, 256^2   (today's code: the reference runs
#                                           in the production checkout are
#                                           from ff354c6, 25 Sep)
#              hlld_ai       64^2, 256^2   one-shot, AI_CKPT
#              hlld_ai ws    64^2, 256^2   warm start (network seed + HLLD's
#                                           secant), same checkpoint
#            all with --face-stats.  Threads: 16 per 256^2 run, 5 per 64^2.
#   compare  scripts/rotor_compare.py --csv: each run against the 512^2 HLLD
#            reference (REF, read-only), and hlld_ai / warm start against
#            hlld at the same resolution
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD-pstar
#   sbatch scripts/calea_pstar_rotor.sh
#   STAGES=compare sbatch --time=0:30:00 scripts/calea_pstar_rotor.sh
#
# Knobs: STAGES ("proof runs compare"), OUT (results/pstar_rotor), AI_CKPT
# (checkpoints/union/v3_s1.pt; its stats are <ckpt stem>_norm_stats.npz),
# REF (the production checkout's results/rotor_512_hlld), RESOLUTIONS
# ("64 256"), MODES ("hlld ai ws"), PY, HLLD_DIR, RMHD_DIR, BASELINE.
# DRYRUN=1: preflight only.
set -u -o pipefail
BASE=/mnt/rafast/miler/codes/rotor2d
HLLD_DIR=${HLLD_DIR:-$BASE/HLLD-pstar}
RMHD_DIR=${RMHD_DIR:-$BASE/rmhd_final}
BASELINE=${BASELINE:-/mnt/rafast/miler/baseline}
PY=${PY:-$HOME/venv/rmhd/bin/python}
OUT=${OUT:-results/pstar_rotor}
STAGES=${STAGES:-"proof runs compare"}
AI_CKPT=${AI_CKPT:-checkpoints/union/v3_s1.pt}
AI_STATS=${AI_STATS:-${AI_CKPT%.pt}_norm_stats.npz}
REF=${REF:-$BASE/HLLD/results/rotor_512_hlld}
RESOLUTIONS=${RESOLUTIONS:-"64 256"}
MODES=${MODES:-"hlld ai ws"}
cd "$HLLD_DIR" || { echo "PREFLIGHT FAILED: no $HLLD_DIR"; exit 2; }
export PYTHONPATH=$HLLD_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

has() { case " $STAGES " in *" $1 "*) return 0 ;; esac; return 1; }
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no python $PY"; exit 2; }
for f in "$AI_CKPT" "$AI_STATS" scripts/run_2d.py scripts/rotor_compare.py; do
    [ -f "$f" ] || { echo "PREFLIGHT FAILED: missing $f"; exit 2; }
done
grep -q '"hlld_ai": None' scripts/run_2d.py || { echo "PREFLIGHT FAILED: this run_2d.py has no hlld_ai"; exit 2; }
git rev-parse --verify -q origin/main >/dev/null || { echo "PREFLIGHT FAILED: no origin/main"; exit 2; }
if has proof; then
    [ -f "$RMHD_DIR/scripts/gate.sh" ] || { echo "PREFLIGHT FAILED: no $RMHD_DIR/scripts/gate.sh"; exit 2; }
    [ -f "$BASELINE/sweep_base_njit_1.pt" ] || { echo "PREFLIGHT FAILED: no recording in $BASELINE"; exit 2; }
fi
if has compare; then
    [ -f "$REF/snap_fin.npz" ] || { echo "PREFLIGHT FAILED: no reference $REF/snap_fin.npz"; exit 2; }
fi
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: stages '$STAGES' ckpt=$AI_CKPT res='$RESOLUTIONS' modes='$MODES' out=$OUT"; exit 0; }

mkdir -p "$OUT"
echo "== $(date)  host=$(hostname)  HLLD $(git rev-parse --short HEAD) ($(git branch --show-current))  main=$(git rev-parse --short origin/main)  stages: $STAGES"
"$PY" -c "import numpy, torch, src; print('numpy', numpy.__version__, 'torch', torch.__version__, 'src', src.__file__)"
md5sum "$AI_CKPT" "$AI_STATS"

# each child's own status (a bare `wait` returns 0 even when children died)
waitall() {
    local bad=0 p
    for p in "$@"; do wait "$p" || { echo "  process $p FAILED ($?)"; bad=1; }; done
    return $bad
}

if has proof; then
    echo "== proof  $(date +%T)"
    MAIN=$OUT/main_tree
    rm -rf "$MAIN"; mkdir -p "$MAIN"
    git archive origin/main | tar -x -C "$MAIN" || { echo "git archive failed"; exit 1; }
    P=""
    common="--problem rotor --n 64 --tend 0.05 --nsnap 4 --log-every 1 --restart-every 0 --solver hlld"
    ( cd "$MAIN" && PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TORCH_THREADS=8 \
        "$PY" -u scripts/run_2d.py $common --out "$HLLD_DIR/$OUT/proof_main" ) \
        > "$OUT/proof_main.log" 2>&1 & P="$P $!"
    TORCH_THREADS=8 "$PY" -u scripts/run_2d.py $common --out "$OUT/proof_branch" \
        > "$OUT/proof_branch.log" 2>&1 & P="$P $!"
    TORCH_THREADS=8 "$PY" -u scripts/run_2d.py $common --face-stats --out "$OUT/proof_branch_tally" \
        > "$OUT/proof_branch_tally.log" 2>&1 & P="$P $!"
    waitall $P || { echo "PROOF FAILED: a run crashed"; exit 1; }
    "$PY" - "$OUT/proof_main" "$OUT/proof_branch" "$OUT/proof_branch_tally" <<'PYEOF' || { echo "PROOF FAILED: main and branch differ"; exit 1; }
import glob, os, sys
import numpy as np
ref, *others = sys.argv[1:]
ok = True
snaps = sorted(os.path.basename(f) for f in glob.glob(os.path.join(ref, "snap_*.npz")))
for o in others:
    same_diag = open(os.path.join(ref, "diag.csv"), "rb").read() == open(os.path.join(o, "diag.csv"), "rb").read()
    nbad = narr = 0
    for f in snaps:
        a, b = np.load(os.path.join(ref, f)), np.load(os.path.join(o, f))
        assert sorted(a.files) == sorted(b.files), (f, a.files, b.files)
        for k in a.files:
            x, y = a[k], b[k]
            narr += 1
            eq = (x.view(np.uint64) == y.view(np.uint64)).all() if x.dtype.kind == "f" else np.array_equal(x, y)
            nbad += int(not eq)
    rows = sum(1 for _ in open(os.path.join(ref, "diag.csv"))) - 1
    print(f"{o} vs {ref}: diag.csv ({rows} steps) {'IDENTICAL' if same_diag else 'DIFFERS'}; "
          f"{len(snaps)} snapshots, {narr} arrays, {nbad} differ in their bits")
    ok &= same_diag and nbad == 0 and len(snaps) > 0
print("BITWISE IDENTICAL" if ok else "DIFFERENCES FOUND")
sys.exit(0 if ok else 1)
PYEOF
    grep -h "^done in\|^rotor" "$OUT"/proof_*.log
    echo "== gate bitwise  $(date +%T)"
    R=$RMHD_DIR H=$HLLD_DIR PY=$PY BASE=$BASELINE bash "$RMHD_DIR/scripts/gate.sh" bitwise ||
        { echo "PROOF FAILED: gate.sh bitwise"; exit 1; }
fi

if has runs; then
    echo "== runs  $(date +%T)"
    P=""; TAGS=""
    for n in $RESOLUTIONS; do
        thr=$([ "$n" -ge 256 ] && echo 16 || echo 5)
        for m in $MODES; do
            case "$m" in
                hlld) args="--solver hlld" ;;
                ai)   args="--solver hlld_ai --ai-checkpoint $AI_CKPT --ai-norm-stats $AI_STATS --ai-mode oneshot" ;;
                ws)   args="--solver hlld_ai --ai-checkpoint $AI_CKPT --ai-norm-stats $AI_STATS --ai-mode warmstart" ;;
                *) echo "unknown mode $m"; exit 2 ;;
            esac
            tag=${m}_$n
            TORCH_THREADS=$thr "$PY" -u scripts/run_2d.py --problem rotor --n "$n" \
                --tend 0.4 --nsnap 8 $args --face-stats --out "$OUT/$tag" \
                > "$OUT/$tag.log" 2>&1 & P="$P $!"; TAGS="$TAGS $tag"
            echo "  $tag: $thr threads ($args)"
        done
    done
    rc_runs=0; set -- $TAGS
    for p in $P; do
        t=$1; shift
        if wait "$p"; then echo "  $(date +%T) $t done: $(tail -n 1 "$OUT/$t.log")"
        else st=$?; rc_runs=1; echo "  $(date +%T) $t FAILED (exit $st):"; tail -n 5 "$OUT/$t.log" | sed 's/^/    /'; fi
    done
    [ "$rc_runs" = 0 ] || { echo "RUNS FAILED"; exit 1; }
fi

if has compare; then
    echo "== compare  $(date +%T)"
    CSV=$OUT/l1.csv; rm -f "$CSV"
    for n in $RESOLUTIONS; do
        for m in $MODES; do
            [ -f "$OUT/${m}_$n/snap_fin.npz" ] || { echo "  no $OUT/${m}_$n"; continue; }
            echo "-- ${m}_$n vs 512^2 reference"
            "$PY" scripts/rotor_compare.py "$OUT/${m}_$n" "$REF" --csv "$CSV"
        done
        for m in ai ws; do
            [ -f "$OUT/${m}_$n/snap_fin.npz" ] && [ -f "$OUT/hlld_$n/snap_fin.npz" ] || continue
            echo "-- ${m}_$n vs hlld_$n"
            "$PY" scripts/rotor_compare.py "$OUT/${m}_$n" "$OUT/hlld_$n" --csv "$CSV"
        done
    done
fi
echo "== $(date)  done"
