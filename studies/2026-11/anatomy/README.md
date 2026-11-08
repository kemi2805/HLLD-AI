# Anatomy of the four-quadrant 128² exact run (ft3), 2026-11-04

User: "I need an analysis on what faces succeed and fail. and why the fail. And
what faces use the 5 wave solver and why. Be ambitious".

Run: calea 77226, `results/r2d_128_exact_ml_r2d_it3` (HLLD-harvest = 51e9c67 +
harvest-cap patch, rmhd 9ca5a4b, primary network ft3, retries big s42/s47), 738
steps to t = 0.4, every exact face harvested.

## Data built

- `face_table.py` (calea 78405, 34 s): every attempted face with its sweep,
  time, position, final outcome, the seven-wave refusal reason, both states,
  the answer (seven-wave and planar faces) and ~70 physics features.
  - Mapping: within a sweep the solved rows are the coverage's seven-wave faces
    then its planar faces, in face order; the unsolved rows are the faces the
    seven-wave path refused, in face order, except 5,656 lanes refused after
    the harvest (degenerate answers of reclassified lanes) which have no row.
    The planar answers' unsolved twins (bitwise equal states) anchor the
    alignment; 2,968 faces (0.06%) stay ambiguous.
  - Checks: 259/259 shards consistent, 393,944/393,944 twins bitwise, outcome
    counts = run_meta exactly (seven 4,257,806, planar 393,944, four-unknown
    25,114, lost 37,940; 12,446 degenerate answers have no row).
  - Output on calea: `HLLD/results/r2d_retrain/anatomy128/faces/faces_*.npz`
    (858 MB); `sample.npz` = every non-seven face + every retry + 400k first-try
    + 200k planar, with weights back to the run (local copy in data/).
- `conditioning.py` (laptop, 2,000 faces per group from six solved shards):
  production's 6x6 seven-wave Jacobian (fullfuncd6) and the planar 3x3 at the
  known answers.
- seed ceiling (HLLD scripts/seed_ceiling.py, calea 78408; small radii 78415
  with `seed_ceiling_fine.py`, whose radius label keeps 0 / 3e-4 / 1e-3 apart --
  78408's %.2f merged 0, 1e-3, 3e-3).
- failure ledger on the lost faces (calea 78406 check, 78407: 1,500 "lost" +
  600 solved; replay: 872 truly failing, 628 four-unknown/degenerate answers,
  0 of 600 controls lost), methods a, b, c, d; no tube label (9-17 h/shard).
- `anatomy_figs.py` -> figs/: atlas, timeline, physics, regimes, mechanism.

## Findings

Attempted faces (4,714,804 with rows): seven-wave first try 89.6%, after a
retry 0.71%, planar five-wave 8.36%, four-unknown 0.53%, lost 0.80%.

**Two regimes hold every rescue and every loss** (state features only):

| regime | faces | lost | planar | four-unknown | retry |
|---|---|---|---|---|---|
| A: slow-Alfven gap < 1e-3, field oblique (\|Bn\|/\|B\| <= 0.9) | 2.20M | 0.87% | 14.4% | 0 | 0.6% |
| B: field near-normal (\|Bn\|/\|B\| > 0.9) | 0.51M | 3.4% | 14.9% | 4.6% | 3.9% |
| neither | 1.99M | 0.000% | 0.03% | 0 | 0.05% |

**The planar rescue (8.4%).**
- Where: the 0.99c shear layers along both axes, the outline of the dense lens,
  the lower-left roll-up.
- States: slow-Alfven gap median 4e-5 (first try 2-4e-3), beta 17.5, tangential
  Lorentz factor 3.6 (half above 3).
- Answers: a slow wave within 1e-4 of the Alfven speed in 94% (first try 33%);
  a slow SHOCK there in 76% (17%). Among faces whose answer has a slow shock at
  the Alfven speed, 29% need the planar solver, 79% at W_t 5.5-6.5.
- Not the seed: the answers are seven-wave roots (99.2% <= 1e-8, all
  constructible); ft3's seed is close (median error 0.014) yet converges on
  3.7%.  From the known answer moved by a random step of 3e-4: 35%; 1e-3: 21%;
  3e-3: 12.5%; 0.01: 8%; 0.1: 5% (first try 99.5-99.9% up to 1e-3, 95% at
  0.1).  ft3's own error shrunk x0.5 / x0.25 / x0.1: 13 / 24 / 37%.  The
  basin is below 3e-4 on two thirds of the planar faces.
- Why: at the answer, production's seven-wave Jacobian cannot be built on 52%
  of planar faces (a 1e-6 probe leaves the constructible set both ways; 70% in
  the gap band 3e-5..1e-4), and on the rest its fifth singular value is
  7.5e-7 of the first -- at the Newton's 1e-6 truncation -- against 1.7e-2 on
  first-try faces.  The planar 3x3 at the same answers: 1.4e-2, always
  buildable.  Reading: with the slow shock at the Alfven speed the Alfven
  rotation and the slow wave act on the same field; the rotation angle and the
  slow wave's two prescribed tangential components become dependent, and the
  shock family ends at the Alfven speed so the probe falls off it.  The
  planar system has no rotation unknown and gives each slow wave one number.

**Seven-wave after a retry (0.7%).** Fast-Alfven near-degeneracy (near-normal
field at beta < 1) and field reversals; ft3's seed error 0.22 there (0.007 on
first-try faces) -- the retry networks, trained on generic data, do better.

**The four-unknown rescue (0.53%).** Near-normal field (99.6% above 0.9),
strong field (beta < 1 in 82%, sigma 2.6), 15% reversal, slow-Alfven NOT
degenerate (gap 3e-2): the tangential field is the wrong slow-wave coordinate
(docs 5a), and the four-unknown solver steers by pressure.

**Lost (0.80%).**
- A-type (53% of losses): the 0.99c shear (93% at W_t > 5.5), no reversal,
  beta > 1, jump 5%; in bursts along the axis lines at t < 0.1 and 0.2-0.3.
- B-type (47%): near-normal field, no shear, 22% reversal, jump 7.6%, later in
  the run (roll-up, jump head); loss rate 6-8% above 5% jumps.
- Production's verdicts on the 872 ledger lanes: seven-wave not converged 94%,
  self-crossing 6%; planar rescue not converged 95%, fails the full residual
  5%.
- Ledger verdict (78407, done 13:11; 872 truly lost lanes, controls find
  another root on <= 1.8%): R1 64 (7.3%), R2 12 (1.4%), R3 74 (8.5%),
  R4 722 (82.8%) -- as 10-28's 64^2 ledger (R4 82.7%).
  - A-type (423): R1 16, R3 72 (17%, nearly all by the intermediate-shock
    branch d), R4 335 (79%); no compound labels.  Where a seven-wave root
    exists it is mostly a slow shock past the Alfven speed, which
    production's window excludes.
  - B-type (449): R1 48, R2 12, R3 2, R4 387 (86%); compound rule 22%;
    recoveries spread over a/b/c/d.
  - So ~83% of the losses have no root any method here finds: a new
    formulation, not more search.

**Predictability.** Gradient boosting on the two states, trained t < 0.3,
tested t >= 0.3: planar vs seven AUC 0.90 (slow-Alfven gap first), lost vs
rescued 0.96, four-unknown vs planar 0.98, retry vs first try 0.84.

## Examples of lost faces (figs/lost_examples.png, 2026-11-04)

Each face's own Riemann problem as a 1D tube (tubes_examples.py, calea 78450:
HLLD + MC at 1024/2048/4096 cells, t = 0.18), ledger lanes 1556 (solved,
planar) 62, 458 (lost, A) 1790 (solved, seven) 1099, 215 (lost, B).  Mirror
faces across the diagonal carry bitwise the same problem.

- A (0.99c shear line): every slow and Alfven wave sits within 0.02 of x/t = 0,
  on the face itself.  Lane 62: between the right slow (0.004) and Alfven
  (0.013) speeds |B_t| dips 0.378 -> 0.318 and recovers to 0.365, converged
  in resolution.  |B_t| changes AT the Alfven speed, which a coplanar Alfven
  wave cannot do: read as a slow shock running at the Alfven speed with a slow
  fan attached behind it -- a compound slow wave WITHOUT field reversal, two
  slow-family waves on one side, outside the seven-wave family.
  compound_wave.py's element requires a reversal, so it cannot build this.
- B (field nearly normal): the field reverses through zero across ONE sharp
  jump at the coincident slow/Alfven speed of the left state (lane 1099), or
  twice, once per side, although the two states have the same sign (lane 215):
  intermediate shocks.  The ledger's intermediate branch found roots for only
  2 of 449 B faces, so the branch as built does not reach these.
- HLLD's flux on the two states vs the tube's state at x/t = 0: lost faces
  3e-4 .. 6e-3 (max relative); the solved neighbours 6e-4 and 1.5e-2.
