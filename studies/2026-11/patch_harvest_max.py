"""Raise the harvest caps for the long four-quadrant runs (2026-11-03).

run_2d.py hard-codes max_solved = max_unsolved = 5,000,000 under --harvest-all,
which a 512^2 run passes early -- after that it records nothing.  This adds
--harvest-max and --harvest-flush (defaults unchanged, so every other run is
untouched) and passes them through calea_rotor_exact.sh as HARVEST_MAX and
HARVEST_FLUSH.  Only what is recorded changes, never the solution.

Usage: python3 patch_harvest_max.py WORKTREE   (an HLLD checkout at 51e9c67)
"""
import sys

wt = sys.argv[1]


def sub(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70])
        s = s.replace(old, new)
    open(path, "w").write(s)


sub(f"{wt}/scripts/run_2d.py", [
    ('''                         "coverage map is always recorded")
''',
     '''                         "coverage map is always recorded")
    ap.add_argument("--harvest-max", type=int, default=None,
                    help="cap on the solved and on the unsolved rows "
                         "(default 5,000,000 with --harvest-all, else the "
                         "Harvester's 200,000); a 512^2 run passes 5M early")
    ap.add_argument("--harvest-flush", type=int, default=None,
                    help="rows per harvest shard (default: the Harvester's "
                         "20,000)")
'''),
    ('''                hk.update(only_retried=False, max_solved=5_000_000,
                          max_unsolved=5_000_000)
''',
     '''                hk.update(only_retried=False, max_solved=5_000_000,
                          max_unsolved=5_000_000)
            if a.harvest_max is not None:
                hk.update(max_solved=a.harvest_max,
                          max_unsolved=a.harvest_max)
            if a.harvest_flush is not None:
                hk.update(flush_every=a.harvest_flush)
'''),
])

sub(f"{wt}/scripts/calea_rotor_exact.sh", [
    ('''# PY, HLLD_DIR, TAU_WEAK (1e-2: the weak-jump gate), TAG (a suffix for OUT
# and LOG).''',
     '''# PY, HLLD_DIR, TAU_WEAK (1e-2: the weak-jump gate), TAG (a suffix for OUT
# and LOG), HARVEST_MAX (the cap on solved and on unsolved rows; 5M) and
# HARVEST_FLUSH (rows per shard; 20k).'''),
    ('''$PY -c "import numpy, numba, torch; print('numpy', numpy.__version__, 'numba', numba.__version__, 'torch', torch.__version__)"
''',
     '''echo "== harvest: all=$HARVEST_ALL  max=${HARVEST_MAX:-(default)}  flush=${HARVEST_FLUSH:-(default)}   tree $(pwd)$(git diff --quiet HEAD -- . || echo ' WITH UNCOMMITTED CHANGES')"
$PY -c "import numpy, numba, torch; print('numpy', numpy.__version__, 'numba', numba.__version__, 'torch', torch.__version__)"
'''),
    ('''    $( [ "$HARVEST_ALL" = 1 ] && echo --harvest-all )
''',
     '''    $( [ "$HARVEST_ALL" = 1 ] && echo --harvest-all ) \\
    ${HARVEST_MAX:+--harvest-max "$HARVEST_MAX"} ${HARVEST_FLUSH:+--harvest-flush "$HARVEST_FLUSH"}
'''),
])
print("patched", wt)
