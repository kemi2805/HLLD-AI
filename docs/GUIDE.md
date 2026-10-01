# A guide to the code and the physics

Written 2026-09-29. This is the entry document: it explains the physics the
two repositories implement, how the code is organised, how a flux is actually
computed, and what has been measured so far. It is meant to be read top to
bottom once, and then used as a map.

The two repositories:

| repository | package | what it is |
|---|---|---|
| `rmhd_final` | `rmhd` | the **exact Riemann solver** for special-relativistic MHD, its batched kernels and its ML warm start |
| `HLLD` | `src/` | the **finite-volume code** (1D and 2D) that uses that solver as a flux function, plus every study built on it |

Where the detail lives once you have the map:

| question | document |
|---|---|
| what is solved, how often, and why the rest is not | `docs/what_we_solve.md` (the long one; every number measured) |
| the numerical scheme, line by line | `docs/numerical_setup_notes.md`, `numerical_setup.tex` |
| the five-wave planar solver | `docs/planar_solver_notes.md`, `planar_solver.tex` |
| the 1D validation tests | `docs/tests.tex` |
| measured ranges of the interface states | `docs/rotor_envelope.md`, `ot_envelope.md` |
| the scalar solver's flow | `rmhd_final/docs/flow_diagram.md` |

---

## Part 1 — The physics

### 1.1 The equations

Ideal special-relativistic magnetohydrodynamics (SRMHD): a perfectly
conducting fluid in flat spacetime. Units: c = 1, and the magnetic field
absorbs the factor 1/sqrt(4π).

**Primitive variables** — what we think in:
`rho` (rest-mass density), `p` (gas pressure), `v = (vx, vy, vz)` (velocity),
`B = (Bx, By, Bz)` (magnetic field in the lab frame).

**Derived quantities**

    W   = 1 / sqrt(1 - v²)                 Lorentz factor
    h   = 1 + eps + p/rho                  specific enthalpy
    p   = (gamma - 1) rho eps              ideal gas  (gamma = 5/3 for the rotor)
    b⁰  = W (v·B)                          the field in the fluid frame...
    bⁱ  = Bⁱ/W + b⁰ vⁱ
    b²  = B²/W² + (v·B)²                   ...and its invariant strength
    P_tot = p + b²/2                       TOTAL pressure
    sigma = b² / (rho h)                   magnetisation

`P_tot` matters more than `p` here: it is the total pressure, not the gas
pressure, that is continuous across a contact, and it is what the exact
solver carries in its state vector.

**Conserved variables** — what the code evolves:

    D   = rho W
    Sⁱ  = (rho h + b²) W² vⁱ - b⁰ bⁱ
    tau = (rho h + b²) W² - P_tot - (b⁰)² - D
    Bⁱ

They obey conservation laws `∂t U + ∂i Fⁱ(U) = 0`, with one constraint that
is not an evolution equation: **div B = 0**. Keeping it satisfied numerically
is the job of constrained transport (section 2.4).

Going from primitive to conserved is algebra. Going back
("conservative-to-primitive", c2p) is a nonlinear root-find, done in every
cell at every stage (section 2.6).

### 1.2 Seven waves

Linearise the equations in one direction (say x) and you find **seven**
characteristic speeds — seven ways a small disturbance can travel:

| wave | symbol | count | what it carries |
|---|---|---|---|
| fast magnetosonic | LF, RF | 2 | compression; changes rho, p, v, and the magnitude of B_t |
| Alfvén | LA, RA | 2 | a **rotation** of the tangential field and velocity; rho and p unchanged |
| slow magnetosonic | LS, RS | 2 | compression, with B_t changing in the opposite sense to the fast wave |
| contact (entropy) | CD | 1 | a jump in density only; moves with the fluid |

`B_t = (By, Bz)` is the field tangential to the direction of propagation and
`B_n = Bx` the normal one. In 1D, `B_n` is a constant (that is div B = 0).

The speeds are ordered

    LF ≤ LA ≤ LS ≤ CD ≤ RS ≤ RA ≤ RF

The contact moves at `vx`. The Alfvén speeds have a closed form,

    lambda_A = (bˣ ± sqrt(rho h + b²) uˣ) / (b⁰ ± sqrt(rho h + b²) u⁰),   u = W(1, v)

and the four magnetosonic speeds are the roots of a **quartic**, solved in
`rmhd/wave_speeds.py` and `rmhd/batched/wave_speeds_b.py` (Ferrari's method,
`quartic_ferrari.py`).

**Degeneracies.** The ordering uses ≤, not <, and the equalities are where
the trouble is:

- `B_n → 0`: the Alfvén and slow speeds collapse onto the contact. Only the
  two fast waves and a tangential discontinuity remain — three waves.
- `B_t → 0`: the Alfvén speed coincides with the fast or the slow speed.
- In between, speeds can be *nearly* equal, which is worse than exactly
  equal: the system is nominally seven-wave and atrociously conditioned.

On the rotor the Alfvén and slow speeds differ by a median of 5.5e-3, against
2.8e-1 between fast and Alfvén. That closeness is a recurring character in
everything below.

### 1.3 The Riemann problem

Two constant states, left `L` and right `R`, separated at x = 0 at t = 0.
What happens next? The equations have no length scale, so the solution
depends only on `xi = x/t` — it is **self-similar** — and consists of the
seven waves fanning out from the origin with constant states between them:

          t
          |   LF   LA  LS   CD   RS  RA   RF
          |     \   \   \   |   /   /   /
          |      \   \   \  |  /   /   /
          |  L  R2 R3  R4 | R5  R6  R7  R
          +------------------------------------ x
                          0

Eight regions: `L = R1`, six intermediate states `R2 … R7`, and `R = R8`.
The code calls these **zones**. A finite-volume scheme needs exactly one thing
from this picture: the state on the ray `xi = 0`, because the flux through
the cell face is the flux of that state. That is Godunov's method.

### 1.4 What each wave is

**Fast and slow waves** are either a **shock** or a **rarefaction**,
depending on whether the pressure behind is higher or lower than ahead.

- A *shock* is a discontinuity. The states on its two sides are tied by the
  Rankine–Hugoniot jump conditions (conservation across the front). Given the
  state ahead and one number (the pressure behind), the state behind and the
  shock speed follow. Code: `rmhd/shock_speed.py`, `postshock.py`,
  `slow_shock.py`; batched `shock_b.py`, `slow_shock_b.py`.
- A *rarefaction* is a smooth fan. The state behind follows by integrating an
  ODE through the fan. Code: `rmhd/rarefaction.py`, `rarefaction_b.py`.

One jump relation is worth remembering because it decides what a wave can do
to the tangential field:

    rho₁ (u₁² - a₁²) B_t1 = rho₂ (u₂² - a₂²) B_t2

(`u` the flow speed in the shock frame, `a` the Alfvén speed). A slow shock
can reduce `B_t` to zero but cannot reverse it; an Alfvén wave can rotate it
but cannot change its size. Reversing the field *through* zero while
compressing takes something else — section 1.8.

**Alfvén waves** are rotational discontinuities. Non-relativistically they
rotate `B_t` and `v_t` by an angle and change nothing else. Relativistically
the rotation is coupled to the velocity and the solve is a small nonlinear
system of its own (`rmhd/alfven_solver.py`, `alfven_b.py`), with more than
one root — which root is taken is a recurring source of subtlety.

**The contact** carries a density jump only. Across it, `vx`, `P_tot`, the
tangential velocity and the tangential field are all continuous (when
`B_n ≠ 0`).

### 1.5 What "solving" means

The intermediate states are not known in advance. The solver **guesses a few
numbers**, builds the wave fan from both sides inward, and checks whether the
two halves meet at the contact.

The production formulation has **six unknowns**

    unk6 = [ ln p_LF,  ln|Bt_CD|,  psi_CD,  ln p_RF,  phi_L,  phi_R ]

- `p_LF`, `p_RF`: the total pressure behind the left and right fast wave;
- `|Bt_CD|`, `psi_CD`: magnitude and angle of the tangential field at the contact;
- `phi_L`, `phi_R`: the rotation angles of the two Alfvén waves;

and **six residuals**

    fvec6 = [ [[vx]], [[vy]], [[vz]], [[P_tot]], slack_L, slack_R ]

The first four are the mismatch of velocity and pressure across the contact.
The two *slacks* exist because each slow wave is told both components of the
tangential field to reach, while a slow wave has only one free parameter; the
slack measures how far it missed.

A **Newton iteration** drives the residuals to zero. This project's
definition, used everywhere:

> A flux is **exact** if and only if the state behind it satisfies the full
> seven-wave jump conditions to **1e-8**, verified per interface, *and* the
> fan is admissible (its waves are in order).

Not "the solver converged", not "it returned success". The answer is checked
against the full residual `fullfuncv6` and only then counted. Two caveats
learned the hard way:

- *A zero residual is not a solution.* The jump conditions do not order the
  waves: a fan whose Alfvén wave sits on the wrong side of its slow wave can
  have a perfect residual and be meaningless. Hence the wave-order gate
  (`_self_crossing` in `exact_flux.py`).
- *A residual is not a state tolerance.* A residual of 1e-8 pins the state to
  only about 1e-4 in this formulation (condition number ~8e3).

### 1.6 Coplanarity, and the planar solver

A 2D flow keeps velocity and field in the plane. Seen from any cell face, the
tangential fields on the two sides are then **parallel or antiparallel** —
*coplanar*. On the rotor this is true to a median of 7.6e-11.

In that geometry an Alfvén wave can do only two things: nothing, or reverse
the field (a rotation by π). The three angle unknowns lose their meaning, and
the six-unknown system becomes ill-posed in exactly the directions it no
longer needs.

The **five-wave planar solver** (`rmhd/batched/planar5_b.py`) is the
formulation that fits: rotate the frame so the left tangential field points
along +y, and solve

    unk3 = [ ln p_LF,  Bt_CD (signed),  ln p_RF ]      against
    fvec3 = [ [[vx]], [[vt]], [[P_tot]] ]

Three unknowns, three equations, no angles, no slacks. `Bt_CD` is signed and
enters linearly, so a field reversal through the contact is an ordinary sign
change. A π rotation of an Alfvén wave is a discrete **branch** (`flip`), not
a variable. Its answers are accepted only after passing the *full* seven-wave
residual, so they are exact in the same sense as any other.

On the rotor, 93.5% of solved interfaces carry exactly five waves (both fast,
both slow, the contact); the Alfvén waves are silent.

**The four-unknown planar solver** (`rmhd/batched/planar4_b.py`, 2026-09-30)
is the same five-wave picture with one strength per wave. The five-wave
solver gives both slow waves ONE field to reach, and decides shock or fan by
whether |B_t| has to fall. Where the field is nearly normal to the face and
the Alfvén speed exceeds the sound speed, that is the wrong coordinate: there
the fast waves move the tangential field and the slow waves move the
pressure, so the target the slow wave is given hardly responds to it — on
one traced interface a slow shock even *strengthens* the field. Half of what
production lost on the rotor (695 of 1,371 interfaces) sits in that regime.
So

    unk4 = [ ln p_LF,  s_L,  s_R,  ln p_RF ]      against
    fvec4 = [ [[vx]], [[vt]], [[B_t]], [[P_tot]] ]

with `s` the signed field (kind B) or `ln P_tot` (kind P) behind the slow
wave, chosen per wave from the tangent of the slow family at the state it
starts from. Its certificate is its own: every shock's full
Rankine–Hugoniot conditions, the contact, the order of the fan. (The
seven-wave residual rebuilds the waves through the field and fails right
answers on exactly these interfaces.) It starts from slow waves of zero
strength and needs no seed. It runs last, only on faces every other rescue
lost. Production default since 2026-10-01 (`RMHD_PLANAR4=0` turns it off),
as are the planar ladder, the crossed offer, the fast-shock edge scan and
the linearised flux below the gate: together 93.97% → 97.63% of attempted
interfaces exact on held-out windows, nothing lost.

### 1.7 Degenerate structures

When a wave family disappears (section 1.2), the interface is classified
before any solve (`rmhd/batched/classify.py`):

| class | meaning | who solves it |
|---|---|---|
| `FULL7` | seven waves | the six-unknown Newton |
| `COPLANAR` | no rotation; rank-3 Jacobian | six-unknown Newton with fixed rank, or planar |
| `ALFVEN_DEGEN` | Alfvén speed = fluid speed (B_n → 0) | three-wave "p-method" |
| `SLOW_DEGEN`, `FAST_DEGEN` | Alfvén merged with slow / fast | three-wave p-method |

The three-wave solver (`contact_b.py`) has one unknown, the contact pressure.
Thresholds are relative to the width of the wave fan, not absolute.

### 1.8 Compound waves and non-uniqueness

Some interfaces need the tangential field to **reverse through zero while the
flow is compressed**. No elementary wave does that (section 1.4). What nature
— or at least the PDE — does is an **intermediate shock**, or a *compound
wave*: an intermediate shock with a rarefaction attached to it.

Three measured facts (`what_we_solve.md` §5):

1. These structures are **not in the seven-wave family**. No seed and no
   amount of retrying finds them, because the root does not exist there.
2. Coplanar ideal MHD is **non-unique**. Balsara's test 1 has two exact
   solutions, a compound wave and a single intermediate shock, 7.7e-4 apart.
3. A finite-volume scheme **selects** one of them — the intermediate shock —
   and neither the order of the reconstruction (2nd to 7th) nor the flux
   function (HLLD or HLLE) changes the selection. It is made by the scheme's
   dissipation, which is not an ideal-MHD quantity.

On the rotor, compound interfaces are 2.3–2.5% of the attempted ones, and
they are predictable from the states alone: the weaker side's `|B_t|` is
below 0.3 `|B_n|` **and** the tangential field turns by more than π/2.

### 1.9 Why strong magnetisation is hard

At high `sigma` the gas pressure is a small difference of large numbers
(`p = P_tot - b²/2`) and the wave speeds crowd together near the speed of
light. The basin from which Newton converges shrinks. Convergence is
*threshold-shaped* in the quality of the seed: within about 0.10 of the root
it converges ~88% of the time, and a seed that is merely somewhat better
gains nothing. This is why *more starts* (multistart, an ensemble of seeds)
has repeatedly helped where *a better seed* has not.

---

## Part 2 — The numerical method

Everything here is in `HLLD/src/physics/`.

### 2.1 Finite volumes

The domain is cut into cells; each holds the cell average of the conserved
variables. A cell changes only through the fluxes across its faces:

    dU/dt = - (F_{i+1/2} - F_{i-1/2}) / dx  -  (G_{j+1/2} - G_{j-1/2}) / dy

In 2D the code is **unsplit**: x- and y-fluxes are computed from the same
state and added. One "sweep" is the computation of all face fluxes in one
direction (`sweep.py`). Each face is a 1D Riemann problem in the direction
normal to it.

### 2.2 Reconstruction

Cell averages are constant per cell; the face needs a left and a right state.
Reconstruction builds them (`reconstruction.py`):

| name | order | note |
|---|---|---|
| `pcm` | 1 | no reconstruction; the original Godunov scheme |
| `mc`, `minmod` | 2 | piecewise linear with a slope limiter; `mc` is **production** |
| `weno5z` | 5 | essentially non-oscillatory |
| `mp5`, `mp7` | 5, 7 | monotonicity-preserving (Suresh & Huynh) |

Two details matter: the code reconstructs `W v` rather than `v`, so the face
velocity is subluminal by construction; and where a reconstructed state would
violate the density or pressure floor, that face falls back to `pcm`. The
stencil width sets the number of ghost cells (`ghosts_needed`: 2, 3, 3, 4).

### 2.3 Riemann solvers — the flux functions

| solver | waves it keeps | code |
|---|---|---|
| HLLE | 2 (the outermost) | `hlld.py::hlle_flux` |
| HLLC | 3 (+ contact) | `hlld.py` |
| **HLLD** | 5 (+ Alfvén) | `hlld.py::hlld_flux` (Mignone, Ugliano & Bodo 2009) |
| **exact** | all 7 | `exact_flux.py::exact_flux_batched` |

Relativistic HLLD is not closed-form: the star pressure `p*` is the root of a
scalar residual, found by a **clamped secant** (the function is called
`safe_secant_bisection` for historical reasons; it does no bisection), about
3.5 iterations to 1e-10. Newton–Raphson was measured against it and is both
slower and less reliable, because the residual has several roots. Where the
root-find fails or its waves come out in the wrong order, that face falls
back to HLLE.

### 2.4 Constrained transport

`ct.py`, after Gardiner & Stone (2005). The in-plane field lives on cell
**faces** (`Bxf`, `Byf`), not at centres, and is updated from an electric
field `Ez` at cell **corners**:

    dBx/dt = - dEz/dy        dBy/dt = + dEz/dx

Because the same `Ez` enters the two faces that share a corner with opposite
signs, the discrete divergence is preserved to round-off — not reduced, not
cleaned: preserved. Measured `|div B| ~ 1e-13`. The Riemann solver supplies
the EMF (the flux of `By` in x is `-Ez`), which is why the flux function's
return contract includes the magnetic fluxes.

### 2.5 Time integration

Strong-stability-preserving Runge–Kutta, third order (`driver2d.py`,
`rk_step_ct`): three stages per step, each a full x-sweep and y-sweep — six
sweeps per step. CFL number 0.25.

### 2.6 Conservative to primitive

`c2p.py`: the robust scheme of Kastaun et al. (2021), a bracketed 1D
root-find per cell. Cells where it fails are counted (`c2p_bad` in
`diag.csv`) and floored.

### 2.7 The magnetic rotor

The production problem (`initial_data2d.py::magnetic_rotor`): a dense
cylinder (`rho = 10`, radius 0.1) spinning rigidly (`omega = 9.95`, so the rim
moves at 0.995 c, `W ~ 10`) in a light ambient medium (`rho = 1`) at uniform
pressure `p = 1`, threaded by a uniform field `B = (1, 0, 0)`. Box
`[-0.5, 0.5]²`, outflow boundaries, run to `t = 0.4`, `gamma = 5/3`.

The field winds up and brakes the rotor, launching fast and slow waves into
the ambient medium. The solution is invariant under a rotation by π about the
axis — exactly, in the continuum — which gives a free measure of numerical
error (`sym_err`).

---

## Part 3 — The exact solver (`rmhd_final`)

### 3.1 Two implementations

| | where | used for |
|---|---|---|
| **scalar** | `rmhd/*.py` | one problem at a time; the reference; `solve_riemann` and the 13 built-in cases |
| **batched** | `rmhd/batched/*.py` | many interfaces at once; what a 2D run uses |

The batched kernels **recompile the scalar source text** at import
(`srcport.py`), so the two cannot drift apart silently. The hot paths also
have numba ports (`*_njit.py`), selected by environment variables
(`RMHD_FAN`, `RMHD_ALFVEN`, `RMHD_SLOWSHOCK`, `RMHD_SHOCK`; production values
in `scripts/kernels.env`).

A warning about the scalar entry point: `solve_riemann` uses an older
*four*-unknown formulation whose residual does not include the tangential
field at the contact. It could report success on a state that was not a
solution (Balsara 5 was wrong by 1.3%). It now measures the contact defect
and refuses. **For a reference solution, use the batched solver.**

### 3.2 The state vector

Everywhere in the solver a state is seven numbers, in the frame of the sweep:

    [ rho, P_tot, vx, vy, vz, By, Bz ]        and B_n = Bx separately

Note `P_tot`, not `p`. `exact_flux.to_solver_frame` converts from the 2D
code's primitives (rotating so the sweep direction is x) and
`from_solver_frame` converts back.

### 3.3 The batched Newton

Entry point: `rmhd.batched.api.make_solver(gamma)` returns a function

    res, diag = solve(left, right, Bn, seed6=..., keys=..., accuracy=1e-8,
                      max_iter=40, n_retries=2, seeds_extra=...)

Inside (`fullcontact_b.py`):

1. `fullfuncv6(left, right, unk6, Bn)` builds the fan — fast wave, Alfvén
   wave, slow wave on each side — and returns the residual, the six zones
   and the wave speeds.
2. The **Jacobian** is a finite difference, but the six perturbed unknown
   vectors are stacked into the batch, so one call replaces seven.
3. The step comes from an **SVD** with small singular values truncated (the
   rank is fixed in advance for coplanar lanes), followed by a **line
   search** that halves the step until the residual drops.
4. Lanes that converge are frozen; the rest continue. *Every lane's
   arithmetic is bit-for-bit what it would have been alone* — the
   batch-independence invariant, which the tests assert.

`res` holds per lane: `unk`, `zones`, `VsLv`, `VsRv` (speeds), `converged`,
`nrm`, `cls`, `attempts`, `n_iter`.

### 3.4 Reading off the flux

`ray_b.py::state_at_xi` finds which region contains `xi = 0` and returns the
state there — a constant zone, or a point inside a rarefaction fan, which
needs its own small integration.

### 3.5 The ML warm start

Newton needs a starting point, and a poor one is why most failures happen.
`rmhd/ml_guess.py` trains a plain MLP (`GuessMLP`) to predict the six
unknowns from the two states.

- **Symmetry first.** The problem is invariant under a rescaling of density,
  pressure and field, a rotation about x, and a reflection. The inputs are
  put in a canonical form (`canonicalize_states`) so the network only ever
  sees physically distinct problems: 13 inputs.
- **Outputs**: 9 numbers — three log-magnitudes, and three angles as
  (cos, sin) pairs, because an angle is discontinuous at ±π as a raw number.
- **Best-of-K**: a head can emit K candidates trained winner-take-all.
- **Checkpoints** (`rmhd_final/data/*.pt`): `ml_guess_rotor_ft.pt` (103k
  parameters, the primary), `ml_guess_rotor_big_s42/s47.pt` (25M parameters,
  used as an ensemble).
- **Retries** (`ml_b.solve_with_retries`): a lane that fails is retried from
  another checkpoint's seed, then from jittered seeds. The jitter is derived
  from a hash of the lane's own features, so a retry cannot depend on which
  other lanes happen to be in the batch.

Training data is **constructed forward** (`riemann_dataset.generate_dataset`):
pick a left state, apply the seven waves one after another, and the result is
a Riemann problem whose exact answer is known by construction. Harvested
rotor solutions are added (`scripts/harvest_to_dataset.py`).

The rule for promoting a checkpoint: **a paired win on a rotor replay at
equal retry budget**, not a validation loss and not the generic eval set —
the two have disagreed.

---

## Part 4 — How a flux is actually computed

`HLLD/src/physics/exact_flux.py::exact_flux_batched`. One call handles every
face of one sweep. The design principle: **HLLD everywhere first, then
overwrite the faces the exact solver earns.** A face that fails any gate
keeps a valid flux.

    all faces of the sweep
      │
      ├─ HLLD flux on every face                          (the safety net)
      │
      ├─ GATES — most faces stop here
      │     bad      superluminal or non-positive state
      │     weak     relative jump < 1e-2: exact and HLLD agree to O(jump²)
      │     upwind   the whole fan lies on one side: HLLD is already exact
      │
      ├─ ATTEMPTED faces (about 12% of face-evaluations on the rotor)
      │     ML seed  ->  seven-wave Newton, up to 3 rounds x 40 iterations
      │     accept if: converged, ray resolved, state physical, fan ordered
      │
      ├─ RESCUES on the faces the seven-wave path lost
      │     five-wave planar solver   (RMHD_PLANAR5_FALLBACK, on since 2026-10-01;
      │                                 before that by each launcher's environment)
      │       + six more starts, and the self-crossing refusals
      │                                 (RMHD_PLANAR5_LADDER / _CROSSED, on)
      │     three-wave solver         (off: it is not exact)
      │     four-unknown planar solver, last  (RMHD_PLANAR4, on)
      │
      ├─ the faces BELOW the weak-jump gate: the linearised solver
      │                                 (RMHD_WEAK_FLUX=linear, on; never "exact")
      │
      ├─ FLUX = flux of the state at xi = 0, where a solver succeeded
      │
      └─ VERIFY against the full seven-wave residual (1e-8), count, record

What production gets, per attempted face-evaluation on the 64² rotor (the
first run with per-face provenance, 2026-09-28):

| answered by | share |
|---|---|
| seven-wave Newton | 43.9% |
| planar five-wave rescue | 49.6% |
| degenerate class | 1.1% |
| **fell back to HLLD** | **5.4%** |

**Diagnostics.** After every call `LAST_DIAG` (a dict in `hlld.py`) holds the
counts and per-lane masks. A run writes `diag.csv` (one row per logged step),
`snap_*.npz` (snapshots), `restart.npz` and `run_meta.json` (what made the
run: scheme, grid, git revision, cost).

**The harvest** (`harvest.py`, `--harvest DIR`) records solved and unsolved
interfaces as training data, and **coverage bitmaps**: for every face of every
sweep, whether it was attempted, whether it came out exact, which solver
answered, and why a face was never tried. `scripts/plot_rotor_output.py`
turns those into the solver map.

**The worker pool** (`exact_pool.py`, `RMHD_POOL`) spreads the seven-wave
solve over processes. A sweep's wall time is set by its *slowest lane*, which
is always a failing one running its full budget — not by the number of cores.

---

## Part 5 — Running things

Python environment on the laptop: `/Users/miler/Codes/py-work/bin/python`.

```bash
# the 2D rotor with HLLD (minutes)
python scripts/run_2d.py --problem rotor --n 64 --solver hlld

# ... with another reconstruction
python scripts/run_2d.py --problem rotor --n 64 --solver hlld --limiter weno5z

# pictures of a run: the state, and the solver map if it has a harvest
python scripts/plot_rotor_output.py results/rotor_64_exact --out figs/rotor_output

# compare two runs (B is the reference)
python scripts/rotor_compare.py results/rotor_64_hlld results/rotor_128_hlld --regions
```

**The exact solver is heavy** (about 30 ms per interface; a 64² run takes
~7.6 hours on a 64-core node, against 4 minutes for HLLD). It runs on the
cluster, never on the laptop:

```bash
ssh itp
cd /mnt/rafast/miler/codes/rotor2d/HLLD
N=64 sbatch scripts/calea_rotor_exact.sh          # exact flux
N=128 LIMITER=mc sbatch scripts/calea_rotor_hlld.sh   # approximate flux
```

Rules that exist because something went wrong without them: submit through
Slurm from `itp`, never by ssh to a node; one full node at a time; on Goethe
only through `rmhd_final/scripts/sbatch_checked.sh`; run a small check job
before a long one.

**Environment variables** are read once, at import of `exact_flux.py`. The
ones that matter: `RMHD_ML_CKPT`, `RMHD_ML_CKPTS` (checkpoints),
`RMHD_PLANAR5_FALLBACK` (the planar rescue), the kernel selectors, and
`RMHD_POOL`. The full table is in the README.

---

## Part 6 — How we know it is right

**The bitwise gate** (`rmhd_final/scripts/gate.sh bitwise`, ~35 s): recorded
interface states are replayed through the solver and the output must be
identical *to the bit*. It protects against the dangerous kind of edit — not
one that crashes, but one that quietly changes every flux.

**The suites** (`gate.sh full`): pytest on both repositories, run across a
node. Baselines: HLLD 149 passed / 7 expected failures; rmhd_final 173 passed
/ 2 skipped / 3 known failures on calea. The expected failures are strict and
documented; an unexpected pass is also a failure.

**Nothing is committed before the gates pass.**

Habits the project has paid for:

- *Paired comparisons only.* Two full runs differ for two reasons — a
  different answer on the same input, or a different input because an
  earlier step differed. `scripts/rotor_replay.py` records sweeps and replays
  identical inputs, which separates the two. Counts are compared with an
  exact McNemar test.
- *Gate before counting.* Un-gated brute force once doubled an apparent
  solve rate with meaningless self-crossing fans.
- *Compare runs from one machine.* The rotor is chaotic: the same code on two
  hosts agrees to 1e-15 early and drifts to 1e-4 by t = 0.4.
- *A run that goes non-finite must fail.* `run_2d.py` once reported a run as
  finished with every cell NaN; it now aborts. Trust the exit status and
  `run_meta.json`, not the existence of `snap_fin.npz`.
- *State the null first.* Several of the results below are negative, and
  were only believable because the expectation was written down beforehand.

---

## Part 7 — What has been measured

Each line is a result with its section in `what_we_solve.md`.

| finding | where |
|---|---|
| The rotor is a **five-wave, coplanar** flow; the Alfvén waves are silent on 93.5% of solved interfaces. | §2 |
| Production solves **94.6%** of the interfaces it attempts; half of those through the planar rescue. | §4 |
| Only ~12% of face-evaluations are attempted; the rest are weak jumps or upwind fans. | §4b |
| Switching on the exact flux changes the rotor by **0.5%**, against a 32.8% gap between resolutions. The exact solver was never the limit on accuracy. | §7 |
| The **reconstruction** moves the rotor 24× more than the exact flux. WENO5-Z beats the production scheme by 10–12% at 256²; MP5 and MP7 lose, break positivity, and go non-finite at 512². | §7 |
| The HLLD star pressure is best found by the **secant**; every Newton variant is slower and falls back ten times as often. | §7 |
| About a third of the unsolved interfaces are **compound** — not in the seven-wave family — and predictable from two numbers. The rest are crowded wave speeds. | §5 |
| Ideal coplanar MHD is **non-unique**; the scheme's dissipation selects the intermediate shock, regardless of reconstruction order or flux function. | §5 |
| Skipping predicted-compound interfaces costs 6.1% of the exact fluxes for 8% of the time. Not worth it. | §5 |
| A sign error in the Alfvén speeds for `B_n < 0` had been costing 18% of the exact fluxes. Fixed. | §5 |
| More training data for the seed: two null results. **Multistart** is the lever that works. | memory notes |

**In progress (science plan H, "toward 100% solved").** The aim is to raise
the 94.6%. First a *failure ledger*: every failing interface of a fixed set of
recorded sweeps is offered to every method available — seven-wave with many
diverse seeds, planar over all flip branches, the ε-continuation ladder, a
seed read off a 1D tube simulation, and the intermediate-shock branch — each
answer held to production's acceptance. That table is the ceiling. Machine
learning is then aimed at the interfaces that have a root we fail to find,
and an extension of the solution family at the ones that do not.

---

## Part 8 — File index

**`rmhd_final/rmhd/`**

| file | what |
|---|---|
| `eos.py` | equation of state (ideal; Meliani) |
| `wave_speeds.py`, `poly_solvers.py` | the seven characteristic speeds |
| `shock_speed.py`, `postshock.py`, `slow_shock.py` | shocks |
| `rarefaction.py` | rarefaction fans |
| `alfven_solver.py` | the rotational discontinuity |
| `contact.py`, `fullcontact.py` | three-wave and seven-wave solves (scalar) |
| `solver.py` | `solve_riemann`, the scalar entry point |
| `riemann_dataset.py` | forward-constructed training data |
| `ml_guess.py` | the network, training, checkpoints |
| `batched/api.py` | **the batched entry point**, `make_solver` |
| `batched/classify.py` | structure classes |
| `batched/fullcontact_b.py` | the six-unknown Newton |
| `batched/planar5_b.py` | the five-wave planar solver |
| `batched/planar4_b.py` | the four-unknown planar solver: one strength per wave, its own certificate |
| `batched/contact_b.py` | the three-wave solver |
| `batched/ray_b.py` | the state at xi = 0 |
| `batched/ml_b.py` | batched prediction and the retry ladder |
| `batched/*_njit.py`, `srcport.py` | compiled kernels and how they are built |

**`HLLD/src/physics/`**

| file | what |
|---|---|
| `grid.py`, `state.py` | grid, ghost cells, boundary conditions, the state container |
| `initial_data2d.py` | the rotor and Orszag–Tang |
| `reconstruction.py` | PCM, PLM, WENO5-Z, MP5, MP7 |
| `hlld.py` | HLLE, HLLC, HLLD and the primitive/conserved algebra |
| `exact_flux.py` | **the hybrid exact flux** |
| `exact_pool.py` | the worker pool |
| `harvest.py` | training rows and coverage bitmaps |
| `sweep.py`, `driver2d.py` | one direction's fluxes; the RK step |
| `ct.py` | constrained transport |
| `c2p.py` | conservative to primitive |
| `tube_seed.py` | 1D tube simulations of single interfaces |

**`HLLD/scripts/`** — runs (`run_2d.py`, `calea_*.sh`), analysis
(`rotor_compare.py`, `recon_study.py`, `plot_rotor_output.py`,
`harvest_summary.py`), and studies (`snapshot_census.py`,
`compound_classifier.py`, `tube_features.py`, `coplanar_limit.py`,
`compound_planar.py`, `compound_wave.py`, `pstar_methods.py`,
`rotor_replay.py`).

---

## Glossary

| term | meaning |
|---|---|
| **attempted** | a face that passed the gates and was offered to the exact solver |
| **branch / flip** | a π rotation at an Alfvén wave, chosen discretely in the planar solver |
| **coplanar** | tangential fields on both sides parallel or antiparallel |
| **compound wave** | an intermediate shock with a rarefaction attached |
| **exact** | full seven-wave residual ≤ 1e-8, verified, and admissible |
| **face-evaluation** | one face in one sweep; a run has six sweeps per step |
| **harvest** | interfaces recorded during a run, as training data and as coverage maps |
| **intermediate shock** | a shock that reverses the tangential field; non-evolutionary in ideal MHD |
| **lane** | one interface inside a batch |
| **multistart** | retrying a failed solve from different seeds |
| **P_tot** | total pressure, `p + b²/2` |
| **seed** | the Newton's starting guess |
| **sigma** | magnetisation, `b²/(rho h)` |
| **star state** | the state between the waves, next to the contact |
| **sweep** | all face fluxes in one direction |
| **zone** | one of the eight constant regions of the fan |
