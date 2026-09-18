# HLLD — 1D/2D special-relativistic MHD with an exact Riemann flux

A finite-volume SRMHD code (PyTorch tensors, CPU): PLM reconstruction,
SSP-RK2/3, constrained transport in 2D, and four flux functions — HLLE, HLLC,
HLLD (Mignone et al.), and **`exact`**, the Godunov flux from the exact
Riemann solution of the companion repository **rmhd_final** (package `rmhd`,
ML warm-started, verified to a 1e-8 seven-wave residual; interfaces it cannot
resolve fall back to HLLD and are counted). The production problem is the
magnetic rotor.

## Install

```bash
pip install -e <rmhd_final checkout>   # the solver; required only for --solver exact
pip install -e .                       # optional: scripts put this repo on sys.path themselves
```

`rmhd` is deliberately not a declared dependency (it is not on PyPI, and a
same-named stranger could be). Its checkpoints live in the rmhd checkout's
`data/`, so it must be an editable install. The Goethe venvs have no pip:
`VIRTUAL_ENV=<venv> uv pip install -e <rmhd_final>`.

## Entry points

| command | what |
|---|---|
| `python scripts/run_2d.py --problem rotor --n 64 --solver exact` | the 2D driver (rotor, Orszag–Tang); `--help` for the exact-flux knobs, restart, harvest |
| `python scripts/shocktube_compare.py <configs/*.yaml> --solvers hlld,exact` | 1D: runs any YAML config through `src/physics/driver.run`, and scores exact vs HLLD against the analytic profile |
| `python scripts/balsara5_slowwave.py`, `make_slow_shock_tube.py`, `plot_balsara5.py` | the 1D slow-wave tests of `docs/tests.tex` |
| `python scripts/rotor_replay.py record\|replay\|compare` | record the rotor's sweeps and replay them through the solver bitwise — the consolidation's physics gate (`rmhd_final/scripts/gate.sh bitwise`) |
| `scripts/calea_rotor_exact.sh`, `calea_brute.sh`, `calea_rescue.sh` | calea jobs: **`sbatch` from `ssh iota`**, never nohup on calea01/02 |
| `scripts/goethe_rotor_exact.sbatch` | Goethe job: submit through `rmhd_final/scripts/sbatch_checked.sh` |
| `scripts/rotor_compare.py`, `plot_2d.py`, `harvest_summary.py` | analysis of run outputs in `results/` |
| `scripts/brute_force_stubborn.py`, `rescue_collect.py`, `solve_interfaces.py`, `measure_planar_claims.py` | studies of the interfaces the solver fails on |

`configs/`: 27 YAML 1D tests (Balsara 1–5, Komissarov, generic Alfvén,
hydro, Mignone ST/RW, the slow-shock tube). The 2D problems are set by
`run_2d.py`'s `--problem`.

## Environment

Read once, at import of `src/physics/exact_flux.py`, unless noted.

| variable | default | effect |
|---|---|---|
| `RMHD_ML_CKPT` | `data/ml_guess_gamma53_v5.pt` | primary warm-start checkpoint, relative to the rmhd checkout. The default is the shock-tube best (90.7% vs 84.0%); **rotor runs set `data/ml_guess_rotor_ft.pt`** (88.8% vs 83.8%) — deliberate, see the comment above `_DEFAULT_CKPT` |
| `RMHD_ML_CKPTS` | empty | comma-separated extra checkpoints seeding retries 1, 2, … (ensemble multistart) |
| `RMHD_PLANAR5_FALLBACK` | `0` | the five-wave planar rescue; its answers are verified against the full seven-wave residual |
| `RMHD_PLANAR_TOL` / `RMHD_PLANAR5_VERIFY` | `1e-6` / `1e-8` | planarity gate / its verification tolerance |
| `RMHD_HARVEST_PLANAR5` | `1` | harvest planar answers as solved rows |
| `RMHD_3WAVE_FALLBACK` | `0` | reduced fast–contact–fast fallback for `\|B_n\| < RMHD_BN_3WX_MAX` (`0.1`); approximate, hence off |
| `RMHD_VERIFY_EXACT` / `RMHD_VERIFY_TOL` | `1` / `1e-8` | verify every exact flux against the full seven-wave system |
| `RMHD_UPWIND_SKIP` | `1` | skip interfaces whose whole fan is on one side of the ray (flux = upwind state); not bitwise-neutral |
| `RMHD_WAVE_ORDER` / `RMHD_WAVE_ORDER_TOL` | `1` / `1e-6` | reject self-crossing fans (a rotation beyond its slow wave carrying a jump) |
| `RMHD_POOL`, `_THREADS`, `_CHUNKS`, `_MIN` | `0`, cores/P, P, `8` | worker-process pool (`src/physics/exact_pool.py`), off below 2; read per call; bitwise equal to in-process (tests/test_exact_pool.py) |
| `RMHD_FAN`, `RMHD_ALFVEN`, `RMHD_SLOWSHOCK`, `RMHD_SHOCK` | numpy | rmhd's compiled kernels; production sources `rmhd_final/scripts/kernels.env` — see rmhd's README |
| `TORCH_THREADS` | `4` | `run_2d.py`'s torch threads |
| `RETRIES`, `NSTEP`, `REPLAY_OUT` | `0`, `1`, — | `rotor_replay.py` only |
| `HARVEST`, `PROFILE_NPZ`, `ROW` | — | `plot_riemann_profile.py` / `plot_pi_rotation.py` only |

## Tests

`pytest tests` — 108 passed + 7 strict xfails (their causes are in the
markers; an XPASS fails the run). ~5 h serially, so the full suite runs on
calea: `REPO=HLLD sbatch rmhd_final/scripts/calea_pytest.sbatch` from iota
(~65 min at 32 cores). Two tests fail on calea only, identically at every
commit measured; see `rmhd_final/scripts/gate.sh`.

## Docs

| file | status |
|---|---|
| `docs/what_we_solve.md` | **current**: what is solved, how, and how much — every number measured, with its source |
| `docs/numerical_setup.tex` + `_notes.md` | the scheme, with the file:line behind each statement |
| `docs/planar_solver.tex` + `_notes.md` | the five-wave planar solver |
| `docs/tests.tex` | the 1D slow-wave tests and the rotor |
| `docs/rotor_envelope.md`, `ot_envelope.md` | measured interface envelopes (from `data/*_envelope_*.npz`) that set rmhd's training ranges |

`data/ARCHIVE_pstar_surrogate/`: the deleted p\*-surrogate's datasets and
checkpoints (git-ignored, never committed), kept with a README.
