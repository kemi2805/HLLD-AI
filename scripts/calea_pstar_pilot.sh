#!/bin/bash
#SBATCH --job-name=pstar-pilot
#SBATCH --partition=calea
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/rafast/miler/codes/rotor2d/HLLD-pstar/logs/slurm_%x_%j.out
# The p* pipeline's checks and pilot, on one calea node, in stages:
#
#   bitwise  main's hlld_flux against this branch's (with and without the
#            row_diag hook), as raw bytes, on 120k sampled faces in all three
#            orientations incl. B_n = 0 and L = R rows; and a whole 1D ST1
#            run with `solver: hlld` under both code bases.  "main" is
#            `git archive origin/main` of THIS clone, unpacked under $OUT --
#            the production checkout is never touched.
#   tests    the pytest files this branch adds or touches
#   gate     rmhd_final's `gate.sh bitwise` (the exact flux's recorded rotor
#            sweeps, replayed to the bit) with H = this clone
#   data     8 pilot chunks x ROWS rows, both gammas (seed 1, not the
#            production seed), split twice: asinh and plain standardisation
#   train    four short CPU trainings: v2/v3 features x none/asinh
#   eval     those four and the archived checkpoint on the pilot test rows
#   ref      the pressure reference (--ref ptot_avg): two more trainings,
#            two CPU-throughput probes, and an evaluation against the above
#
#   ssh itp; cd /mnt/rafast/miler/codes/rotor2d/HLLD-pstar
#   sbatch scripts/calea_pstar_pilot.sh
#   STAGES="data train eval" ROWS=200000 sbatch scripts/calea_pstar_pilot.sh
#
# Knobs: STAGES, OUT (results/pstar_pilot), ROWS (60000 per chunk), EPOCHS
# (40), ARCHIVED (checkpoints/archived: default-mignone.pt + norm_stats.npz,
# copied from the Mac's data/ARCHIVE_pstar_surrogate -- 1 MB, not the
# dataset), PY, HLLD_DIR, RMHD_DIR, BASELINE.  DRYRUN=1: preflight only.
set -u -o pipefail
BASE=/mnt/rafast/miler/codes/rotor2d
HLLD_DIR=${HLLD_DIR:-$BASE/HLLD-pstar}
RMHD_DIR=${RMHD_DIR:-$BASE/rmhd_final}
BASELINE=${BASELINE:-/mnt/rafast/miler/baseline}
PY=${PY:-$HOME/venv/rmhd/bin/python}
OUT=${OUT:-results/pstar_pilot}
STAGES=${STAGES:-"bitwise tests gate data train eval"}
ROWS=${ROWS:-60000}
EPOCHS=${EPOCHS:-40}
ARCHIVED=${ARCHIVED:-checkpoints/archived}
cd "$HLLD_DIR" || { echo "PREFLIGHT FAILED: no $HLLD_DIR"; exit 2; }
export PYTHONPATH=$HLLD_DIR
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

has() { case " $STAGES " in *" $1 "*) return 0 ;; esac; return 1; }
[ -x "$PY" ] || { echo "PREFLIGHT FAILED: no python $PY"; exit 2; }
for f in scripts/pstar_bitwise_check.py scripts/pstar_make_dataset.py \
         scripts/pstar_build_splits.py scripts/pstar_train.py \
         scripts/pstar_evaluate.py configs/pstar-union.yaml configs/mignone/st1.yaml; do
    [ -f "$f" ] || { echo "PREFLIGHT FAILED: missing $f"; exit 2; }
done
git rev-parse --verify -q origin/main >/dev/null || { echo "PREFLIGHT FAILED: no origin/main in $HLLD_DIR"; exit 2; }
if has gate; then
    [ -f "$RMHD_DIR/scripts/gate.sh" ] || { echo "PREFLIGHT FAILED: no $RMHD_DIR/scripts/gate.sh"; exit 2; }
    [ -f "$BASELINE/sweep_base_njit_1.pt" ] || { echo "PREFLIGHT FAILED: no recording in $BASELINE"; exit 2; }
fi
if has eval; then
    for f in "$ARCHIVED/default-mignone.pt" "$ARCHIVED/norm_stats.npz"; do
        [ -f "$f" ] || { echo "PREFLIGHT FAILED: missing $f"; exit 2; }
    done
fi
[ "${DRYRUN:-0}" = 1 ] && { echo "dry run ok: stages '$STAGES' out=$OUT rows=$ROWS epochs=$EPOCHS"; exit 0; }

mkdir -p "$OUT" logs
echo "== $(date)  host=$(hostname)  HLLD $(git rev-parse --short HEAD) ($(git branch --show-current))  main=$(git rev-parse --short origin/main)  stages: $STAGES"
"$PY" -c "import numpy, torch, src; print('numpy', numpy.__version__, 'torch', torch.__version__, 'src', src.__file__)"
rc=0

# wait for each pid, keep every status (a bare `wait` returns 0 on crashes)
waitall() {
    local bad=0 p
    for p in "$@"; do wait "$p" || { echo "  process $p FAILED ($?)"; bad=1; }; done
    return $bad
}

if has bitwise; then
    echo "== bitwise  $(date +%T)"
    MAIN=$OUT/main_tree
    rm -rf "$MAIN"; mkdir -p "$MAIN"
    git archive origin/main | tar -x -C "$MAIN" || { echo "git archive failed"; exit 1; }
    "$PY" scripts/pstar_bitwise_check.py inputs --out "$OUT/bw_in.npz" || rc=1
    P=""
    PYTHONPATH=$MAIN PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/pstar_bitwise_check.py dump "$OUT/bw_in.npz" --out "$OUT/bw_main.npz" > "$OUT/bw_main.log" 2>&1 & P="$P $!"
    "$PY" scripts/pstar_bitwise_check.py dump "$OUT/bw_in.npz" --out "$OUT/bw_branch.npz" > "$OUT/bw_branch.log" 2>&1 & P="$P $!"
    "$PY" scripts/pstar_bitwise_check.py dump "$OUT/bw_in.npz" --out "$OUT/bw_branch_rd.npz" --row-diag > "$OUT/bw_branch_rd.log" 2>&1 & P="$P $!"
    PYTHONPATH=$MAIN PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/pstar_bitwise_check.py tube configs/mignone/st1.yaml --out "$OUT/tube_main.npz" > "$OUT/tube_main.log" 2>&1 & P="$P $!"
    "$PY" scripts/pstar_bitwise_check.py tube configs/mignone/st1.yaml --out "$OUT/tube_branch.npz" > "$OUT/tube_branch.log" 2>&1 & P="$P $!"
    waitall $P || rc=1
    grep -h "^src =" "$OUT"/bw_*.log "$OUT"/tube_*.log
    "$PY" scripts/pstar_bitwise_check.py compare "$OUT/bw_main.npz" "$OUT/bw_branch.npz" "$OUT/bw_branch_rd.npz" || rc=1
    "$PY" scripts/pstar_bitwise_check.py compare "$OUT/tube_main.npz" "$OUT/tube_branch.npz" || rc=1
fi

if has tests; then
    echo "== tests  $(date +%T)"
    "$PY" -m pytest tests/test_pstar_data.py tests/test_ai_features.py \
        tests/test_ai_modes.py tests/test_idir.py tests/test_envelope.py \
        tests/test_hlld_physics.py tests/test_wave_order.py \
        tests/test_pstar_newton.py tests/test_sweep2d.py \
        -q -n 16 -rfEX -p no:cacheprovider 2>&1 | tail -15 || rc=1
fi

if has gate; then
    echo "== gate bitwise  $(date +%T)"
    R=$RMHD_DIR H=$HLLD_DIR PY=$PY BASE=$BASELINE bash "$RMHD_DIR/scripts/gate.sh" bitwise || rc=1
fi

if has data; then
    echo "== data  $(date +%T)"
    "$PY" scripts/pstar_make_dataset.py --config configs/pstar-union.yaml \
        --out "$OUT/shards" --chunks 0-7 --rows "$ROWS" --workers 8 --seed 1 || rc=1
    P=""
    for tr in asinh none; do
        OMP_NUM_THREADS=8 "$PY" scripts/pstar_build_splits.py --shards "$OUT/shards" \
            --out "$OUT/splits_$tr" --transform $tr > "$OUT/build_$tr.log" 2>&1 & P="$P $!"
    done
    waitall $P || rc=1
    tail -30 "$OUT/build_asinh.log"
fi

if has train; then
    echo "== train  $(date +%T)"
    P=""
    for spec in v2none:splits_none/v2-idir-abs-bn v2asinh:splits_asinh/v2-idir-abs-bn \
                v3none:splits_none/v3-hlld-r-invariants v3asinh:splits_asinh/v3-hlld-r-invariants; do
        name=${spec%%:*}; split=${spec#*:}
        "$PY" -u scripts/pstar_train.py --split "$OUT/$split" --out "$OUT/ck/$name.pt" \
            --device cpu --threads 14 --epochs "$EPOCHS" --batch 4096 \
            > "$OUT/train_$name.log" 2>&1 & P="$P $!"
    done
    waitall $P || rc=1
    for n in v2none v2asinh v3none v3asinh; do echo "  $n: $(tail -1 "$OUT/train_$n.log")"; done
fi

if has eval; then
    echo "== eval  $(date +%T)"
    M=""
    for n in v2none v2asinh v3none v3asinh; do
        M="$M --model $n=$OUT/ck/$n.pt,$OUT/ck/${n}_norm_stats.npz"
    done
    "$PY" scripts/pstar_evaluate.py $M \
        --model archived=$ARCHIVED/default-mignone.pt,$ARCHIVED/norm_stats.npz \
        --testset pilot_test=$OUT/splits_asinh/test_rows.npz \
        --out "$OUT/eval_pilot.json" --workers 48 --chunk 5000 || rc=1
fi

# Second round (2026-10-17): the pressure reference.  Same pilot shards, same
# split seed (so the same test rows), asinh with --ref ptot_avg; the v2/v3
# asinh runs above are the comparison.  Two extra trainings probe the CPU
# throughput for the production job: v3 with ref at batch 16384, and at
# batch 4096 with 16 threads.
if has ref; then
    echo "== ref  $(date +%T)"
    OMP_NUM_THREADS=8 "$PY" scripts/pstar_build_splits.py --shards "$OUT/shards" \
        --out "$OUT/splits_asinh_ref" --transform asinh --ref ptot_avg \
        > "$OUT/build_asinh_ref.log" 2>&1 || rc=1
    P=""
    for spec in "v2ref:v2-idir-abs-bn:4096:14" "v3ref:v3-hlld-r-invariants:4096:14" \
                "v3ref_b16k:v3-hlld-r-invariants:16384:14" "v3ref_t16:v3-hlld-r-invariants:4096:16"; do
        IFS=: read -r name ver batch thr <<< "$spec"
        "$PY" -u scripts/pstar_train.py --split "$OUT/splits_asinh_ref/$ver" \
            --out "$OUT/ck/$name.pt" --device cpu --threads "$thr" --epochs "$EPOCHS" \
            --batch "$batch" > "$OUT/train_$name.log" 2>&1 & P="$P $!"
    done
    waitall $P || rc=1
    for n in v2ref v3ref v3ref_b16k v3ref_t16; do
        echo "  $n: $(tail -n 1 "$OUT/train_$n.log")"
        grep "^epoch  *40 " "$OUT/train_$n.log" | sed 's/^/      /'
    done
    M=""
    for n in v2asinh v3asinh v2ref v3ref v3ref_b16k; do
        M="$M --model $n=$OUT/ck/$n.pt,$OUT/ck/${n}_norm_stats.npz"
    done
    "$PY" scripts/pstar_evaluate.py $M \
        --model archived=$ARCHIVED/default-mignone.pt,$ARCHIVED/norm_stats.npz \
        --testset pilot_test=$OUT/splits_asinh_ref/test_rows.npz \
        --out "$OUT/eval_pilot_ref.json" --workers 48 --chunk 5000 || rc=1
fi
echo "== $(date)  rc=$rc"
exit $rc
