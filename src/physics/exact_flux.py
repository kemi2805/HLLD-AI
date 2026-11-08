"""
Exact Riemann solver as a Godunov flux function.

Drop-in replacement for ``hlld_flux``: same
``flux_fn(primL, primR, eos, idir) -> (F, U, p_star)`` contract, so
``sweep()`` and ``run_2d.py`` need only name it.

This is the SCALAR reference implementation (Phase 1).  It loops over
interfaces and costs ~6 s each, so it is not usable for a production 2D run
-- its purpose is to pin down the physics (frame rotation, the xi = 0 ray,
flux assembly, the fallback contract) and to be the answer the batched
kernels must reproduce.

How a flux is obtained
----------------------
1. Rotate into the sweep frame (normal component first) and convert the gas
   pressure the 2D code carries into the TOTAL pressure the exact solver
   uses.
2. Triage: unphysical or near-uniform interfaces, and every structure class
   the seven-wave solver cannot handle, go straight to HLLD.
3. ML warm start -> ``fullcontact6`` -> the resolved state on the xi = 0 ray.
4. Assemble the flux by calling ``compute_srmhd_fluxes(s*, s*, eos, idir)``
   with the SAME state on both sides: it then returns ``fL == fR == F(U*)``
   and ``uL == uR == U*``, i.e. the full eight-key conserved-flux dict that
   constrained transport reads, computed by the code the rest of the repo
   already trusts.
5. Anything that fails any gate falls back to HLLD, and the fraction is
   reported rather than hidden.

Why the weak-jump gate is not a cop-out
---------------------------------------
Reconstruction leaves ``|q_L - q_R| ~ 1e-8`` across most of a smooth region.
There the exact and linearised fluxes agree to ``O(jump^2)``, so switching is
accuracy-neutral -- while running the exact solver would be both wasteful and
ill-conditioned, since multi-root ranking by Rankine-Hugoniot residual is
meaningless when every residual is at round-off and the finite-difference
Jacobian (EPS = 1e-6) would difference a function whose variation is smaller
than its own discretisation defect.
"""

from __future__ import annotations

import math as _math
import os
import sys

import torch

from .hlld import LAST_DIAG, compute_srmhd_fluxes, hlld_flux

# Overridable: the cluster checkouts live elsewhere (Goethe:
# /work/astro/miler/codes/rmhd_final), and a git worktree of rmhd_final is how
# a bit-identity reference is isolated from edits on main.
# The exact solver is an installed package now (pip install -e <rmhd_final>);
# RMHD_ROOT and the sys.path.append that used to live here are gone.
try:
    import rmhd.paths as _rmhd_paths
except ImportError as _e:                      # pragma: no cover
    raise ImportError(
        "exact_flux needs the rmhd package: pip install -e <rmhd_final checkout>"
    ) from _e

# Which warm-start checkpoint to auto-load.  Override with RMHD_ML_CKPT.
#
# Default is v5 because it is the measured best on the SHOCK-TUBE regime,
# which is what the 1D gate exercises: 90.7% converged with the 4x retry
# ladder, against 89.0% for ml_guess_gamma53.pt (the previous hardcoded
# default) on the same 300-problem eval set.
#
# It is deliberately NOT the right choice for the rotor.  The rotor-trained
# checkpoints win there and lose here, and the gap is significant in both
# directions (paired McNemar, evalset_rotor_600 / evalset_300):
#
#     checkpoint                  rotor    tubes
#     ml_guess_gamma53_v5.pt      83.8%    90.7%
#     ml_guess_rotor_ft.pt        88.8%    84.0%
#     ml_guess_rotor_scratch.pt   87.2%    82.3%
#
# So for rotor production runs, set
#     RMHD_ML_CKPT=data/ml_guess_rotor_ft.pt
# rather than changing this default.  Which one to promote outright is an
# open decision -- ft and scratch are statistically indistinguishable from
# each other on both regimes.
#
# The checkpoint only affects convergence RATE and speed, never correctness:
# interfaces the Newton cannot resolve fall back to HLLD and are counted in
# LAST_DIAG.
_DEFAULT_CKPT = os.environ.get("RMHD_ML_CKPT", "data/ml_guess_gamma53_v5.pt")
# Ensemble multistart: extra checkpoints (comma-separated, relative to the rmhd
# checkout, via rmhd.paths.resolve) seed retries 1, 2, ... in turn instead of the
# Gaussian jitter of the primary seed.  Only used when n_retries > 0.
# The reduced three-wave fallback (plan Track 5 B2), off by default.
#
# MEASURED 2026-09-07 on the 64^2 run's harvest, which is what motivates it:
# the Alfven and slow waves of a rotor interface are nearly COINCIDENT --
# |Vs_LA - Vs_LS| has median 3.5e-3 and 5th percentile 3.0e-6 -- so the
# seven-wave system has two almost parallel wave families and is
# ill-conditioned exactly there.  That is why 56% of attempted interfaces go
# unsolved, and why no starting guess rescues them: an exhaustive scan over
# the eight planar angle choices crossed with a 7^3 magnitude grid still
# converges 8.5%.
#
# The reduced fast-contact-fast solver has no such pair and converges on
# 100% of them.  Its price is neglecting the slow waves, measured against
# 4000 interfaces whose exact answer is known:
#
#     |B_n|        star pressure, median error   p90
#     < 0.03                     9.8e-05         8.2e-04
#     0.03 - 0.1                 5.9e-04         3.2e-03
#     0.1  - 0.3                 2.2e-03         1.1e-02
#     > 0.3                      3.8e-03         3.4e-02
#
# so the gate is on |B_n|, defaulting to 0.1 where the error is still below
# one part in a thousand.  This is the regime of the rotor's y-sweeps, which
# is where coverage is worst (24.7% of attempted resolved, against 64.0% on
# the x-sweeps).
_3WAVE_FALLBACK = os.environ.get("RMHD_3WAVE_FALLBACK", "0") not in ("0", "", "off")
_BN_3WX_MAX = float(os.environ.get("RMHD_BN_3WX_MAX", "0.1"))

# The five-wave PLANAR rescue (2026-09-08), off by default.
#
# A 2D flow is planar, and in the plane the seven-wave formulation is badly
# parameterised: `psi_CD` is fixed by geometry (a planar slow wave turns the
# field by a median of 4e-11 radians -- it can only reverse it), and each slow
# wave is asked for BOTH tangential components when its family has one
# parameter, so most trial unknowns fall outside the reachable set and the
# construction fails outright.  Hence 56% of attempted interfaces go unsolved
# and no seed fixes it (an exhaustive angle enumeration crossed with a 7^3
# magnitude grid converges 8.5%).
#
# The planar solver has three unknowns [ln p_LF, signed Bt_CD, ln p_RF]
# against the three contact jumps -- square, no angles, no slacks -- and from
# a TRIVIAL seed it converges on 65.8% of the interfaces the seven-wave solver
# fails and 91.2% of the ones it solves.
#
# It is admitted here because its answer is EXACT, not close: `planar5_b.solve`
# verifies every answer against the FULL seven-wave residual before reporting
# it converged, and on real interfaces that residual has median 1.1e-10.  The
# check is not a formality -- where the true solution has a genuine pi
# rotational discontinuity (4.1% of rotor interfaces) the five-wave family does
# not contain it and the solver can converge to something else; the
# verification is what keeps those out.  Contrast the three-wave fallback
# above, whose answers never reach 1e-6 on the same test, which is why that one
# stays off and this one is worth having.
# On by default since 2026-10-01.  It was "production" before that only by
# the environment of each launcher and replay script (RMHD_PLANAR5_FALLBACK=1);
# the first run with the five rungs as defaults was submitted without it and
# ran with the five-wave rescue OFF -- the four-unknown solver covered those
# lanes, but it was not the configuration measured.  The bitwise gate sets
# it explicitly, so its recording is unaffected.
_PLANAR5_FALLBACK = os.environ.get("RMHD_PLANAR5_FALLBACK", "1") not in ("0", "", "off")
# The planar rescue's LADDER (production default; see the note below _WEAK_FLUX).  The rescue starts the
# planar Newton once, from the mean state, on the unflipped branch.  The
# failure ledger (2026-09-29, 36 sweeps, 1371 failing interfaces) offered
# every failing lane 4 flip branches x 6 seeds under production's own
# acceptance: 192 were solved, none by a seed the rescue uses.  The rungs
# below are that table's greedy cover, in its order -- fixed on the ledger's
# windows, to be judged on held-out ones -- and are tried only on lanes still
# unsolved, so nothing production solves can change:
#     (flip L, flip R, seed)
_PLANAR5_RUNGS = ((False, False, "+L"), (False, False, "+R"),
                  (False, False, "-L"), (False, True, None),
                  (True, False, None), (False, False, "s2p"))
_PLANAR5_LADDER = os.environ.get("RMHD_PLANAR5_LADDER", "1") not in ("0", "", "off")
# Offer the planar rescue the lanes whose seven-wave answer was refused as a
# self-crossing fan (production default; ledger: 16 of 16 then solved).
_PLANAR5_CROSSED = os.environ.get("RMHD_PLANAR5_CROSSED", "1") not in ("0", "", "off")
# The four-unknown planar solver (production default): one strength per
# wave, each slow wave steered by whichever of (B_t, P_tot) moves along its
# family, its own certificate (every shock's Rankine-Hugoniot conditions,
# the contact, the fan's order).  Offered the lanes every rescue above lost.
# See rmhd.batched.planar4_b and docs/what_we_solve.md 5a.
_PLANAR4 = os.environ.get("RMHD_PLANAR4", "1") not in ("0", "", "off")
# Lanes the four-unknown Newton loses from its own start (the mean total
# pressure, slow waves of zero strength, the steering read off that start) are
# offered again from HLLD's star pressure, which every face has already paid
# for: both fast waves to p*, the slow waves weak fans steered by pressure or
# field (_PLANAR4_STAR_STARTS, in this order), then the same starts with a
# central-difference Jacobian (planar4_b.jacobian), each answer held to the
# same acceptance and to a non-decreasing entropy across every shock.  On the
# four-quadrant 128^2 run's lost near-normal faces the default start is 15%
# off in pressure where both fast waves are rarefactions, and a field-steered
# fan cannot carry B_t through zero, which is how the slow fans there reverse
# the field.  Failure ledger (calea 78576): 195 of the 472 lanes the
# loose-fallback retry still loses, the checked ones within 1e-6 of their own
# tubes (HLLD 1e-3..1e-2); 600 controls, one other root.  RMHD_PLANAR4_STAR=1;
# off by default.
_PLANAR4_STAR = os.environ.get("RMHD_PLANAR4_STAR", "0") not in ("0", "", "off")
# (central differences, steering of the two slow waves, start strength), in
# the order they are tried; steering 0 field, 1 pressure, 2 SIGNED field
# (planar4_b.KIND_S, for beta ~ 100 faces with the field nearly normal, whose
# slow waves are nearly Alfven waves and carry B_t through zero: calea 78609,
# 191 of the 249 lanes the other starts lose)
_PLANAR4_STAR_FULL = tuple(
    [(False,) + st for st in (((1, 1), 1e-4), ((1, 1), 1e-3), ((1, 0), 1e-4),
                              ((0, 1), 1e-4), ((0, 1), 1e-3))]
    + [(True,) + st for st in (((1, 1), 1e-4), ((1, 1), 1e-3), ((1, 0), 1e-4),
                               ((0, 1), 1e-4), ((0, 1), 1e-3), ((2, 2), 1e-4))])
# The starts that answered on the failure ledger, each only once: the full
# set costs 1.85-2.9x wall on the 64^2 runs (numpy Newton calls on every lost
# lane, whose per-call overhead the pool does not spread).  RMHD_PLANAR4_STAR_SET
# = lean (default) | full; RMHD_PLANAR4_STAR_ITER caps each attempt's Newton.
_PLANAR4_STAR_LEAN = ((False, (1, 1), 1e-4), (True, (1, 1), 1e-4),
                      (True, (2, 2), 1e-4), (True, (0, 1), 1e-4),
                      (True, (1, 0), 1e-4))
# Then the same pressure steering from a start that puts a WEAK fast wave
# (p* within 2% of its ahead pressure) just past its own ahead pressure
# instead of at p*: next to the fast-Alfven degeneracy the basin in that
# wave's pressure is narrower than p*'s error (lane 858, 2026-11-05: p* 1.7%
# from p_RF; calea 78627, 13 of the 52 near-normal strong-field lanes left).
# Last, both fast waves at their own ahead pressures x (1 + offset): the
# strong-field near-normal faces whose p* is 2-9% from one side (calea 78634:
# 29 of the 41 lanes everything above loses, none with a falling entropy).
# (central, steering, slow start strength, fast start: "near" / "both", offset)
_PLANAR4_STAR_SIDE = ((True, (1, 1), 1e-4, "near", 1e-3),
                      (True, (1, 1), 1e-4, "near", -1e-3),
                      (True, (1, 1), 1e-4, "both", 1e-3),
                      (True, (1, 1), 1e-4, "both", -1e-3),
                      (True, (1, 1), 1e-4, "both", 1e-2),
                      (True, (1, 1), 1e-4, "both", -1e-2))
_PLANAR4_STAR_STARTS = (_PLANAR4_STAR_FULL
                        if os.environ.get("RMHD_PLANAR4_STAR_SET", "lean") == "full"
                        else _PLANAR4_STAR_LEAN)
_PLANAR4_STAR_ITER = int(os.environ.get("RMHD_PLANAR4_STAR_ITER", "25"))
# The admissibility gate: an accepted answer must also be an ADMISSIBLE
# solution of ideal RMHD as far as three cheap tests tell -- no shock or
# rotation lowers the entropy (by more than RMHD_ADMISSIBLE_GATE_DS), every
# fan's characteristic speed spreads in its direction of travel, and every
# wave (fans as their whole xi-range) sits at or after the previous non-zero
# one, slow against fast included, to _WAVE_ORDER_TOL.  Applied where the
# seven-wave, planar and four-unknown solvers accept, so a refused answer is
# offered to the next rung before it falls back to HLLD.  Measured with an
# independent ideal-RMHD verifier (2026-11-05, 256,000 answers of the 4Q
# 128^2 run): 0.55% / 1.99% of seven-wave / planar answers lower the entropy
# at a jump, 0.17% / 1.73% carry a "fan" that compresses, 0.19% / 0.90% a
# slow wave past its fast one; conservation itself holds.  RMHD_ADMISSIBLE_GATE=1;
# off by default.
_ADMISSIBLE_GATE = os.environ.get("RMHD_ADMISSIBLE_GATE", "0") not in ("0", "", "off")
_ADMISSIBLE_GATE_DS = float(os.environ.get("RMHD_ADMISSIBLE_GATE_DS", "1e-9"))
# The four-unknown answers the gate refuses are, on the 4Q 128^2 failure
# ledger, a ROTATION fused into a slow jump: near-normal field, slow and Alfven
# speeds within 1e-9, the jump runs between the Alfven speeds of its two sides
# and lowers the entropy by 1e-9..7e-5.  planar4 has no rotations, and the
# reversal production builds elsewhere (planar5_b._reverse) finds only the
# rotations that reverse B_t in the lab, while about half of these keep its lab
# sign.  RMHD_PLANAR4_ROT=1: every lane still lost after the star starts that
# had a refused answer is solved again from that answer with the coplanar
# rotation (fullcontact_b.coplanar_rotation, closed form, either lab sign) in
# front of the slow wave -- on both sides, the left, the right -- signed-field
# slow waves, central differences, and accepted only when admissible
# (`_admissible_structure`).  Measured on 6 refused lanes: each has exactly one
# admissible combination, a regular seven-wave answer whose flux differs from
# the refused one by 2e-10..2e-6.  Off by default.
_PLANAR4_ROT = os.environ.get("RMHD_PLANAR4_ROT", "0") == "1"
# RMHD_PLANAR4_STAR_BATCH=1: the star starts (and the rotation choices) of all
# lanes in one solve per difference scheme instead of one solve per start --
# the same answers (the solves are batch-independent), less waiting
# (`_planar4_star_batched`).  Off by default.
_PLANAR4_STAR_BATCH = os.environ.get("RMHD_PLANAR4_STAR_BATCH", "0") == "1"
# RMHD_COPLANAR7=late: the planar form of the seven-wave family
# (rmhd coplanar7_b: signed-field and pressure-steered slow waves, the coplanar
# rotation of either lab sign) as a rescue rung between the five-wave planar
# solver and the four-unknown one, from the sweep's own network seeds
# (`_coplanar7_and_ray`).  Inside solve_batch (RMHD_COPLANAR7=1 or =after) it
# took the faces the five-wave solver answers in 4 iterations and cost the 4Q
# 64^2 run 3.7x the control (2026-11-05); here it sees only what that solver
# left.  Off by default.
_COPLANAR7_LATE = os.environ.get("RMHD_COPLANAR7", "0") == "late"
_PLANAR4_ROT_COMBOS = ((True, True), (True, False), (False, True))
# How far the entropy may fall across a shock of a star-start answer:
# K_behind / K_ahead >= 1 - _PLANAR4_STAR_DS (K = p / rho^gamma).  The
# signed-field answers carry field-reversing jumps whose entropy falls by
# 1e-10..2e-5 and whose fluxes still sit 10-300x closer to the faces' own
# tubes than HLLD (calea 78606 tubes, 2026-11-04); 1e-9 refuses them.
_PLANAR4_STAR_DS = float(os.environ.get("RMHD_PLANAR4_STAR_DS", "1e-9"))
# What the faces BELOW the weak-jump gate get: the linearised solver (the
# default) or HLLD (`linearised.py`, exact to second order in the jump where
# HLLD is first order).  Never counted as exact.
_WEAK_FLUX = os.environ.get("RMHD_WEAK_FLUX", "linear").lower()
# What an ATTEMPTED face gets when every exact rung loses it: HLLD, unless its
# relative jump is below RMHD_WEAK_FALLBACK_TAU (1e-2, production's own weak
# gate) -- then the linearised flux, the same one the gate gives such faces
# when it is shut, and 25-120x closer to the exact flux than HLLD there
# (docs/what_we_solve.md 4c).  It matters only when the gate is opened
# (tau_weak below 1e-2, e.g. TAU_WEAK=1e-12): with the production gate every
# attempted face has a jump of at least 1e-2, so nothing changes.  Measured
# before it existed (2026-10-11): with the gate open, the four-quadrant run's
# HLLD faces went from 0.72% to 3.41% of all faces because the weak faces it
# lost fell back to HLLD.  Never counted exact; recorded per face as
# `linear_fallback` in the coverage map.  0 turns it off.  Requires
# RMHD_WEAK_FLUX=linear (with HLLD below the gate the fallback is HLLD too).
_WEAK_FALLBACK_TAU = float(os.environ.get("RMHD_WEAK_FALLBACK_TAU", "1e-2"))
# All five -- the edge scan (rmhd_final shock_b), the ladder, the crossed
# offer, the four-unknown rung and the linearised flux below the gate -- are
# production defaults since 2026-10-01, the user's decision after the paired
# measurements in docs/what_we_solve.md 4c and 5a (together: 93.97 -> 97.63%
# of attempted interfaces exact on held-out windows, lost 0; the linearised
# flux moves the 64^2 rotor by 0.3% and halves nothing but the symmetry
# error).  Each is switched off with the value 0 (RMHD_WEAK_FLUX=hlld).
_PLANAR_TOL = float(os.environ.get("RMHD_PLANAR_TOL", "1e-6"))
_PLANAR5_VERIFY = float(os.environ.get("RMHD_PLANAR5_VERIFY", "1e-8"))
# Verify every exact flux against the full seven-wave system and report the
# count.  On by default: it is one residual evaluation per solved interface,
# a few percent, and it is what licenses the word "exact".
_VERIFY_EXACT = os.environ.get("RMHD_VERIFY_EXACT", "1") not in ("0", "", "off")
_VERIFY_TOL = float(os.environ.get("RMHD_VERIFY_TOL", "1e-8"))
# Skip interfaces whose fan lies entirely on one side of the ray.  HLLD's
# signal speeds are clamped at zero, so `cmin == 0` means every wave is
# right-going and its flux is f_L exactly; `cmax == 0` gives f_R.  The exact
# solution at xi = 0 is then that same upwind state, so solving is pure waste.
#
# MEASURED on 45,650 harvested interfaces of a 64^2 run: the branch fires on
# 29.07% of ATTEMPTED interfaces (14.57% all-right-going, 14.51% all-left),
# and on those |F_exact - F_HLLD| / |F_HLLD| has median 0, p99 0 and max
# 1.9e-16 -- 100% below 1e-14.  On the other 32,378 interfaces NOT ONE falls
# below 1e-14 (median 3.7e-3), so the test separates the two populations
# exactly rather than approximately.
#
# It is not bitwise-neutral: a handful of lanes differ in the last bit, so a
# run with this on is not bit-reproducible against one with it off.  Set
# RMHD_UPWIND_SKIP=0 to recover the old behaviour.
_UPWIND_SKIP = os.environ.get("RMHD_UPWIND_SKIP", "1") not in ("0", "", "off")
# Compound interfaces: a structure the seven-wave family cannot hold, so no
# amount of retrying finds it.  MEASURED on the snapshot census (plan F,
# `docs/what_we_solve.md` section 5; 10,172 interfaces at 64^2 and 16,930 at
# 128^2): a face whose tangential field is weak against the normal one AND
# reverses is compound 81% of the time, and 67-76% of those faces cannot be
# solved by any path we have.  The rule is two numbers, both already in hand
# before the solve, and it transfers between the two resolutions with no
# refitting.  It is OFF by default: it trades a small number of exact fluxes
# (24% of the faces it flags ARE solvable) for the retry budget spent on the
# rest, and that trade has to be measured per run before it is taken.
_COMPOUND_SKIP = os.environ.get("RMHD_COMPOUND_SKIP", "0") not in ("0", "", "off")
_COMPOUND_BT_MAX = float(os.environ.get("RMHD_COMPOUND_BT_MAX", "0.3"))
_COMPOUND_DPSI_MIN = float(os.environ.get("RMHD_COMPOUND_DPSI_MIN", str(_math.pi / 2)))
# Do not offer the four-unknown solver the lanes the compound rule flags
# (opt-in): it certified 0 of 100 tube-labelled compound interfaces and burns
# its full budget on each, serially, in the main process.  Measured on the
# cost replay before deciding.
_PLANAR4_SKIP_COMPOUND = os.environ.get("RMHD_PLANAR4_SKIP_COMPOUND", "0") not in ("0", "", "off")
# Before the planar rungs, drop the lanes whose input is not planar to
# _PLANAR_TOL: no planar rung can accept them (the five-wave rescue, its
# ladder and the four-unknown solver all refuse planar_resid > _PLANAR_TOL,
# with the same `planar5_b.to_planar`), but each runs its Newton first. On a
# non-coplanar problem that is every lane the seven-wave solver loses: the
# four-quadrant run with the field tilted out of the plane offered them all
# to three solves that could only refuse. Skipping them changes no flux,
# mask or star pressure; their refusal reason becomes "not planar" (2) in
# both rungs' codes, without the solve. Default on; 0 restores the old path.
_PLANAR_PRECHECK = os.environ.get("RMHD_PLANAR_PRECHECK", "1") not in ("0", "", "off")
# The continuation rung (src/physics/continuation.py): the seven-wave solve
# walked in from the face tilted by eps = 0.1 down to 1e-6, then polished on
# the real face. Last, and only on lanes that are NOT planar and every other
# rung lost -- where nothing else can answer; on a coplanar problem it is
# never offered a lane, so it changes nothing there. Measured on the
# four-quadrant ledger with the field tilted out of the plane: 63% of the
# losses recovered, another root on 1 of 1,187 controls (docs section 7).
# Opt-in until measured on a whole run.
_CONTINUATION = os.environ.get("RMHD_CONTINUATION", "0") not in ("0", "", "off")
# Solve each face as the canonical member of its symmetry orbit (reflection,
# signed exchanges of the tangential axes, the sign of B) in the x-frame, and
# map the answer back (src/physics/canonical.py).  Mirror-image faces then get
# mirror-image fluxes by construction; without it, 0.4-0.7% of Orszag-Tang's
# mirror pairs had one face solved and the other on HLLD (calea job 40966).
# Production default since 2026-10-13 (the user's decision), after whole runs:
# Orszag-Tang 97.31 -> 97.67% solved, pi-asymmetry 700x smaller at t = 1,
# 1.5x faster; the four-quadrant problem bitwise x <-> y symmetric, 10%
# faster; L1 against the 512^2 references unchanged.  RMHD_CANONICAL=0 turns
# it off.
_CANONICAL = os.environ.get("RMHD_CANONICAL", "1") not in ("0", "", "off")
# The jump conditions do not order the waves.  A fan whose rotational
# discontinuity sits on the far side of its slow wave is self-crossing and is
# not a Riemann solution, however small its residual -- verification cannot
# see this, because it only asks that the jumps be consistent.
#
# A crossing between ZERO-STRENGTH waves is harmless: nothing depends on
# where a wave that carries no jump sits, and that is the common case.
# MEASURED over 24,632 production rows: the left rotation crosses the slow
# wave on 10.0% of interfaces but carries a jump on only 5.6%, so
# **2.52% are genuinely inadmissible** and were being counted as exact.  The
# planar path is unaffected (its rotations are zero by construction: it
# crosses on 89.8% and is inadmissible on 0.03%).
_WAVE_ORDER = os.environ.get("RMHD_WAVE_ORDER", "1") not in ("0", "", "off")
_WAVE_ORDER_TOL = float(os.environ.get("RMHD_WAVE_ORDER_TOL", "1e-6"))
# Harvest the planar answers as solved rows.  On by default WITH the rescue:
# they are exact solutions of interfaces the seven-wave solver cannot reach,
# which is precisely the population the warm-start network has no training
# data for.  `attempts` is set to 2 so they survive a harvester left at its
# `only_retried` default -- these lanes did fail the seven-wave attempt first,
# so that is a description rather than a fiction.
_HARVEST_PLANAR5 = os.environ.get("RMHD_HARVEST_PLANAR5", "1") not in ("0", "", "off")
# Harvest the later rungs' answers too (RMHD_HARVEST_RESCUES, off by default):
# the planar seven-wave rung's as source=2 and the four-unknown solver's as
# source=3, each row with the slow-wave steering of each side, the rotations
# and the stage that answered it (harvest.py: `steer`, `rot`, `stage`).  The
# data for a planar warm start (2026-11-06): on the 4Q 64^2 run the late rung
# answered 12 of its 50 faces from the network seed and 38 only with the
# pressure-steered slow waves, and 34 more faces it could not answer needed
# the four-unknown solver.
_HARVEST_RESCUES = os.environ.get("RMHD_HARVEST_RESCUES", "0") not in ("0", "", "off")

_EXTRA_CKPTS = [c.strip() for c in os.environ.get("RMHD_ML_CKPTS", "").split(",")
                if c.strip()]
# (the append-never-insert dance is gone: `rmhd.eos` cannot shadow `src.physics.eos`)

_V = ("vx", "vy", "vz")
_B = ("Bx", "By", "Bz")
_S = ("Sx", "Sy", "Sz")


def _check_eos(eos, gamma_expected=None):
    """The two repos must agree on the equation of state.

    ``rmhd_final``'s ``enthalpy_scalar`` gives ``h = 1 + gamma p / ((gamma-1) rho)``;
    ``hybrid_eos`` gives ``h = 1 + gamma_th p / ((gamma_th-1) rho)`` plus a cold
    part scaled by ``K``.  They coincide only for ``K == 0`` and
    ``gamma_th == gamma`` -- true for the 2D runs, but solving a different
    problem than intended is exactly the kind of error that produces
    plausible output, so it is asserted rather than assumed.
    """
    K = float(getattr(eos, "K", 0.0))
    g = float(getattr(eos, "gamma"))
    gt = float(getattr(eos, "gamma_th", g))
    if K != 0.0 or abs(gt - g) > 1e-14:
        raise ValueError(
            f"exact_flux requires an ideal gas (K=0, gamma_th==gamma); got "
            f"K={K}, gamma={g}, gamma_th={gt}"
        )
    return g


def _b2(vx, vy, vz, Bx, By, Bz):
    v2 = vx * vx + vy * vy + vz * vz
    W2 = 1.0 / torch.clamp(1.0 - v2, min=1e-14)
    vB = vx * Bx + vy * By + vz * Bz
    return (Bx * Bx + By * By + Bz * Bz) / W2 + vB * vB


def to_solver_frame(s, eos, idir):
    """(rho, P_total, v_n, v_t1, v_t2, B_t1, B_t2) plus B_n, all ``(N,)``."""
    n, t1, t2 = idir, (idir + 1) % 3, (idir + 2) % 3
    p_gas, _ = eos.press_and_cs2(s["eps"], s["rho"])
    b2 = _b2(s["vx"], s["vy"], s["vz"], s["Bx"], s["By"], s["Bz"])
    P_tot = p_gas + 0.5 * b2
    return (s["rho"], P_tot, s[_V[n]], s[_V[t1]], s[_V[t2]],
            s[_B[t1]], s[_B[t2]]), s[_B[n]]


def from_solver_frame(state, Bn, eos, idir, like):
    """Inverse of :func:`to_solver_frame`, back to a primitive dict."""
    n, t1, t2 = idir, (idir + 1) % 3, (idir + 2) % 3
    rho, P_tot, vn, vt1, vt2, Bt1, Bt2 = state
    out = {"rho": rho}
    out[_V[n]], out[_V[t1]], out[_V[t2]] = vn, vt1, vt2
    out[_B[n]], out[_B[t1]], out[_B[t2]] = Bn, Bt1, Bt2
    b2 = _b2(out["vx"], out["vy"], out["vz"], out["Bx"], out["By"], out["Bz"])
    out["p"] = torch.clamp(P_tot - 0.5 * b2, min=1e-30)
    out["eps"] = eos.eps__press_rho(out["p"], out["rho"])
    return out


def relative_jump(sL, sR):
    """Scale-free, rotation-invariant measure of how discontinuous an interface is.

    The obvious form -- max over every variable of |a-b|/(|a|+|b|) -- is wrong
    for any quantity that changes SIGN.  The denominator vanishes at a zero
    crossing, so the ratio saturates at 1 no matter how smooth the data is.
    That is not a corner case here: the rotor is a rotating flow, so vx, vy and
    By cross zero across most of the domain.  Measured on a 128^2 HLLD rotor at
    t=0.4, the old form called 99.9% of interfaces discontinuous at a 1e-2
    threshold, while rho and p -- the only strictly positive variables -- put
    just 3.2% and 7.9% above 0.1.  The weak-jump gate could therefore never
    fire, and every interface went to the exact solver.

    Three fixes, one per kind of variable:

    * rho and p are strictly positive, so the ordinary relative difference is
      well behaved and is kept.
    * velocity is compared as an absolute VECTOR jump.  v is in units of c and
      bounded by 1, so |v_L - v_R| is already dimensionless and scale-free, and
      it cannot blow up where a component passes through zero.
    * the field is compared as a vector jump against a scale floored by the
      pressure scale, so a component crossing zero -- or |B| itself vanishing --
      cannot empty the denominator.

    Using vector norms rather than a per-component max also makes the measure
    rotation-invariant, which the old form was not: the same physical interface
    could gate differently in an x-sweep than in a y-sweep, which would quietly
    break `test_idir` and the rotor's pi-rotation symmetry diagnostic.
    """
    tiny = 1e-30
    worst = (sL["rho"] - sR["rho"]).abs() / (
        sL["rho"].abs() + sR["rho"].abs() + tiny)
    worst = torch.maximum(worst, (sL["p"] - sR["p"]).abs() / (
        sL["p"].abs() + sR["p"].abs() + tiny))

    dv = sum((sL[k] - sR[k]) ** 2 for k in ("vx", "vy", "vz")).sqrt()
    worst = torch.maximum(worst, dv)

    dB = sum((sL[k] - sR[k]) ** 2 for k in ("Bx", "By", "Bz")).sqrt()
    BL = sum(sL[k] ** 2 for k in ("Bx", "By", "Bz")).sqrt()
    BR = sum(sR[k] ** 2 for k in ("Bx", "By", "Bz")).sqrt()
    scale = BL + BR + (sL["p"].abs() + sR["p"].abs()).sqrt()
    return torch.maximum(worst, dB / (scale + tiny))


def _self_crossing(Zc, VL, VR, tol):
    """True where a seven-wave answer's fan crosses itself (see _WAVE_ORDER).

    ``Zc`` is the six zones R2..R7, ``VL``/``VR`` the ``[LF, LA, LS]`` and
    ``[RF, RA, RS]`` speeds: arrays on the batched path, floats on the scalar
    one, which must apply the SAME rule or it is not the batched path's
    reference.
    """
    import numpy as np
    cd = 0.5 * (Zc[2][2] + Zc[3][2])        # contact speed, from R4/R5
    order = [VL[0], VL[1], VL[2], cd, VR[2], VR[1], VR[0]]

    def _rot_strength(A, B):
        rho = np.maximum(np.abs(Zc[A][0]), 1e-30)
        return np.maximum(
            np.abs(Zc[A][0] - Zc[B][0]) / rho,
            np.sqrt(sum((Zc[A][k] - Zc[B][k]) ** 2 for k in (2, 3, 4))))

    # a rotation may sit out of order ONLY if it carries no jump
    crossed = (order[2] - order[1] < -tol) & (_rot_strength(0, 1) > 1e-8)
    crossed = crossed | ((order[5] - order[4] < -tol) & (_rot_strength(4, 5) > 1e-8))
    # every other adjacent pair must be ordered unconditionally
    for _k in (0, 2, 3, 5):
        crossed = crossed | (order[_k + 1] - order[_k] < -tol)
    return crossed


def _admissible_structure(P, gamma, sL, sR, Bn, zones6, VsLv, VsRv):
    """True where an answer passes the admissibility gate (_ADMISSIBLE_GATE).

    ``zones6`` are the six states R2..R7 between the outer states ``sL`` and
    ``sR``, ``VsLv``/``VsRv`` the [fast, Alfven, slow] speeds of each side.
    A wave is a discontinuity where its own speed satisfies the jump
    conditions (planar4_b's scaled residual <= 1e-6), else a fan; a
    rotation weaker than production's 1e-8 (as in `_self_crossing`) and a
    wave of zero strength are skipped.
    """
    import numpy as np
    S4 = P["planar4"]
    xi = P["xi"]
    n = Bn.shape[0]
    st = [list(sL)] + [list(z) for z in zones6] + [list(sR)]
    V = [VsLv[0], VsLv[1], VsLv[2], 0.5 * (zones6[2][2] + zones6[3][2]),
         VsRv[2], VsRv[1], VsRv[0]]

    def K(X):
        rho, Pt, vx, vy, vz, By, Bz = X
        W2 = 1.0 / np.maximum(1.0 - (vx * vx + vy * vy + vz * vz), 1e-300)
        vB = vx * Bn + vy * By + vz * Bz
        return (Pt - 0.5 * ((Bn ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / rho ** gamma

    ok = np.ones(n, dtype=bool)
    lo, hi = np.zeros((7, n)), np.zeros((7, n))
    live = np.zeros((7, n), dtype=bool)
    with np.errstate(all="ignore"):
        for w in range(7):
            left = w < 3
            a, b = (st[w], st[w + 1]) if left else (st[w + 1], st[w])
            Bm = np.sqrt(Bn ** 2 + np.maximum(a[5] ** 2 + a[6] ** 2, b[5] ** 2 + b[6] ** 2))
            sc = [np.abs(b[0]), np.abs(b[1]), 1.0, 1.0, 1.0, Bm, Bm]
            d = np.max(np.stack([np.abs(a[j] - b[j]) / sc[j] for j in range(7)]), axis=0)
            zero = d < 1e-12
            if w in (1, 5):
                dv = np.sqrt(sum((a[j] - b[j]) ** 2 for j in (2, 3, 4)))
                zero |= np.maximum(np.abs(a[0] - b[0]) / np.abs(a[0]), dv) <= 1e-8
            live[w] = ~zero
            if w == 3:
                lo[w] = hi[w] = V[w]
                continue
            rh = S4["rh_residual"](a, b, V[w], Bn)
            disc = ~zero & (rh <= 1e-6)
            ratio = K(b) / K(a) - 1.0
            ok &= ~(disc & np.isfinite(ratio) & (ratio < -_ADMISSIBLE_GATE_DS))
            if w in (1, 5):
                lo[w] = hi[w] = V[w]
                continue
            sw = ("LF", None, "LS", None, "RS", None, "RF")[w]
            fan = ~zero & ~disc
            la, lb = xi(a, sw, Bn), xi(b, sw, Bn)
            spread = (lb - la) if left else (la - lb)
            ok &= ~(fan & np.isfinite(spread) & (spread < -1e-9))
            lo[w] = np.where(fan, np.fmin(la, lb), V[w])
            hi[w] = np.where(fan, np.fmax(la, lb), V[w])
        last = np.full(n, -np.inf)
        for w in range(7):
            behind = live[w] & np.isfinite(lo[w]) & (lo[w] < last - _WAVE_ORDER_TOL)
            ok &= ~behind
            last = np.where(live[w] & np.isfinite(hi[w]), hi[w], last)
    return ok


def exact_flux(sL, sR, eos, idir: int = 0, *, model=None, scaler=None,
               fallback=hlld_flux, tau_weak: float = 1e-6,
               accuracy: float = 1e-8, max_iter: int = 40,
               n_retries: int = 0, nsamples: int = 400):
    """Godunov flux from the exact RMHD Riemann solution at ``xi = 0``."""
    from rmhd.batched import classify as C
    from rmhd.batched import ray as RAY
    from rmhd.batched import ray_scalar as RS

    gamma = _check_eos(eos)
    N = sL["rho"].numel()

    # HLLD everywhere first; the exact solve then overwrites the lanes it owns
    F, U, p_star = fallback(sL, sR, eos, idir=idir)
    F = {k: v.clone() for k, v in F.items()}
    U = {k: v.clone() for k, v in U.items()}
    p_star = p_star.clone()
    p_hll_fb = p_star.abs().clone()

    (L7, BnL) = to_solver_frame(sL, eos, idir)
    (R7, BnR) = to_solver_frame(sR, eos, idir)

    # sweep() forces the normal field to be single-valued at a face; if that
    # contract is broken the whole wave structure is ill-posed, so check it.
    dBn = float((BnL - BnR).abs().max())
    if dBn > 1e-10 * float(BnL.abs().max().clamp(min=1e-30)):
        raise ValueError(f"normal field is not single-valued at the face "
                         f"(max |Bn_L - Bn_R| = {dBn:.3e})")

    jump = relative_jump(sL, sR)
    v2L = sL["vx"] ** 2 + sL["vy"] ** 2 + sL["vz"] ** 2
    v2R = sR["vx"] ** 2 + sR["vy"] ** 2 + sR["vz"] ** 2
    bad = (v2L >= 1.0) | (v2R >= 1.0) | (sL["rho"] <= 0) | (sR["rho"] <= 0)
    weak = (~bad) & (jump < tau_weak)

    # ── structure classification (Level 1) ────────────────────────────────
    la_L, ra_L = C.alfven_speeds(L7[0], L7[1], L7[2], L7[3], L7[4],
                                 L7[5], L7[6], BnL, gamma)
    la_R, ra_R = C.alfven_speeds(R7[0], R7[1], R7[2], R7[3], R7[4],
                                 R7[5], R7[6], BnR, gamma)
    from rmhd.wave_speeds import xi as _xi_scalar

    def _eigs(st, Bn):
        out = torch.zeros(N, 4, dtype=st[0].dtype)
        for i in range(N):
            s = [float(c[i]) for c in st]
            try:
                allv, _, _ = _xi_scalar(s, "LF", float(Bn[i]), gamma,
                                        return_all=True)
                out[i] = torch.tensor([float(v) for v in allv])
            except Exception:
                out[i] = torch.tensor([-1.0, 0.0, 0.0, 1.0], dtype=out.dtype)
        return out

    eL, eR = _eigs(L7, BnL), _eigs(R7, BnR)
    cls, _info = C.classify(L7[2], R7[2], la_L, ra_L, la_R, ra_R, eL, eR,
                            L7[5], L7[6], R7[5], R7[6])

    # Phase 1 solves only the classes fullcontact6 is defined for.  The
    # reduced 3-wave path for the degenerate classes is Phase 5; until then
    # they take the HLLD fallback, which is reported, not hidden.
    solvable = (~bad) & (~weak) & ((cls == C.FULL7) | (cls == C.COPLANAR))

    n_conv = n_ray_fan = n_unphys = n_crossed = 0
    solved_idx: list[int] = []
    idx = torch.nonzero(solvable).reshape(-1).tolist()

    if idx:
        from rmhd import ml_guess as mg
        from rmhd.fullcontact import fullcontact6
        if model is None or scaler is None:
            model, scaler = mg.load(_rmhd_paths.resolve(_DEFAULT_CKPT))

    for i in idx:
        left = [float(c[i]) for c in L7]
        right = [float(c[i]) for c in R7]
        Bn = float(BnL[i])
        try:
            if n_retries:
                res, _ = mg.predict_unk6_retry(
                    model, scaler, left, right, Bn, gamma, accuracy=accuracy,
                    max_iter=max_iter, n_retries=n_retries)
            else:
                u6 = mg.predict_unk6(model, scaler, left, right, Bn)
                res = fullcontact6(left, right, u6, Bn, gamma,
                                   accuracy=accuracy, max_iter=max_iter)
            if res is None or not res["converged"]:
                continue
            # the batched path's admissibility gate, same rule, same tolerance
            if _WAVE_ORDER and _self_crossing(res["zones"], res["VsLv"],
                                              res["VsRv"], _WAVE_ORDER_TOL):
                n_crossed += 1
                continue
            st, reg = RS.state_at_xi(left, right, res["zones"], res["VsLv"],
                                     res["VsRv"], Bn, gamma, xi_target=0.0,
                                     nsamples=nsamples)
        except Exception:
            continue

        rho, P_tot, vn, vt1, vt2, Bt1, Bt2 = st
        v2 = vn * vn + vt1 * vt1 + vt2 * vt2
        if not (rho > 0.0 and v2 < 1.0):
            n_unphys += 1
            continue
        W2 = 1.0 / (1.0 - v2)
        eta = Bn * vn + Bt1 * vt1 + Bt2 * vt2
        b2 = (Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2) / W2 + eta * eta
        if P_tot - 0.5 * b2 <= 0.0:
            n_unphys += 1
            continue

        one = {k: v[i:i + 1] for k, v in sL.items()}
        star = from_solver_frame(
            [torch.tensor([c], dtype=sL["rho"].dtype) for c in st],
            torch.tensor([Bn], dtype=sL["rho"].dtype), eos, idir, one)
        # with the SAME state on both sides this returns uL == uR == U*
        # and fL == fR == F(U*)  -- the full conserved-flux dict CT needs
        uS, _, fS, _, _, _ = compute_srmhd_fluxes(star, star, eos, idir)

        for k in F:
            F[k][i] = fS[k][0]
            U[k][i] = uS[k][0]
        p_star[i] = P_tot
        solved_idx.append(i)
        n_conv += 1
        if reg in RAY.FAN_REGIONS:
            n_ray_fan += 1

    fell_back = N - n_conv
    exact_mask = torch.zeros(N, dtype=torch.bool)
    if solved_idx:
        exact_mask[torch.tensor(solved_idx, dtype=torch.long)] = True
    LAST_DIAG.update(
        exact_mask=exact_mask,
        solver="exact",
        idir=idir,
        n_interfaces=N,
        n_bad=int(bad.sum()),
        n_weak_gate=int(weak.sum()),
        **C.histogram(cls),
        n_attempted=len(idx),
        n_exact=n_conv,
        n_fan_interior=n_ray_fan,
        n_unphysical_star=n_unphys,
        n_wave_order_rejected=n_crossed,
        n_hlld_fallback=fell_back,
        frac_hlld_fallback=fell_back / max(N, 1),
        frac_exact=n_conv / max(N, 1),
    )
    # p_star already carries hlld_flux's negative sentinel on every lane that
    # fell back, because those lanes were never overwritten -- so
    # tests/test_idir.py's branch-flip logic works unchanged.
    return F, U, p_star


# ===========================================================================
# Batched path -- the one a production run uses
# ===========================================================================

_SOLVER_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def _load_model_cached(path):
    """torch.load the warm-start checkpoint once per process, not per flux call.

    Every sweep called mg.load, i.e. a torch.load from disk -- ~800 loads per
    32^2 run, ~9000 at 64^2.  Numerics-neutral: the same file yields the same
    weights.  Keyed by path so RMHD_ML_CKPT can still switch checkpoints.
    """
    if path not in _MODEL_CACHE:
        from rmhd import ml_guess as mg
        _MODEL_CACHE[path] = mg.load(path)
    return _MODEL_CACHE[path]


def _batched_parts(gamma):
    """Build (and cache) the batched solvers and ray for one ``gamma``.

    Cached because ``make_solver`` source-recompiles the scalar kernels, which
    is not something to repeat per sweep.
    """
    if gamma in _SOLVER_CACHE:
        return _SOLVER_CACHE[gamma]
    from rmhd.batched import api as API
    from rmhd.batched import profile as PR
    from rmhd.batched import ray_b as RAY
    import numpy as np

    # The ray sampler: xi is fullcontact_b.make_xi, which asks xi_all's per-ROOT
    # validity about the requested root only.  Near B_n = 0 the interior pair
    # merges into a double root double precision cannot resolve while the fast
    # pair stays exact, and B_n = By IS the rotor's y-sweep -- so gating LF/RF
    # on the interior pair would refuse well-determined rarefactions (rmhd_final
    # commit 2392aa6).  This body used to be spelled out here; it is the same one.
    xi_fn, fan_p, fan_n = PR.make_sampler(gamma)
    from rmhd.batched import contact_b as CB
    from rmhd.batched import planar5_b as P5
    from rmhd.batched import planar4_b as P4
    parts = dict(solve=API.make_solver(gamma), ray=RAY, xi=xi_fn,
                 fan_p=fan_p, fan_n=fan_n, np=np,
                 reduced=CB.make_solver(gamma),
                 planar5=P5.make_solver(gamma),
                 planar4=P4.make_solver(gamma))
    if _COPLANAR7_LATE:
        from rmhd.batched import coplanar7_b as CP7
        parts["coplanar7"] = CP7.make_solver(gamma)
    _SOLVER_CACHE[gamma] = parts
    return parts


def _solve_and_ray(P, gamma, subL, subR, subBn, seed6, keys, seeds_extra, *,
                   accuracy, max_iter, n_retries, tau_bt):
    """The per-lane numerical pipeline of one sweep, on solver-frame arrays.

    Solve (seven-/three-wave, with the retry ladder), then resolve the state
    at xi = 0 per structure class.  Pure numpy/numba on per-lane inputs, so
    it runs unchanged on a slice of the sweep -- which is what
    `exact_pool` does with it on a many-core node.

    Returns ``(res, diag, star, region, ray_ok, n_fan)``.
    """
    np = P["np"]
    RAY = P["ray"]
    n = int(np.asarray(subBn).shape[0])
    res, d = P["solve"](subL, subR, subBn, seed6=seed6, accuracy=accuracy,
                        max_iter=max_iter, n_retries=n_retries, keys=keys,
                        seeds_extra=seeds_extra, tau_bt=tau_bt)
    from rmhd.batched import classify as C
    cls = res["cls"]
    conv = res["converged"]

    # ── xi = 0, by structure class ────────────────────────────────────────
    star = [np.zeros(n) for _ in range(7)]
    ray_ok = np.zeros(n, dtype=bool)
    region = np.full(n, -1, dtype=int)
    n_fan = 0

    m7 = np.flatnonzero(conv & np.isin(cls, (C.FULL7, C.COPLANAR)))
    c7 = np.zeros(n, dtype=bool)
    if "coplanar7" in res:
        # answers of the planar form (RMHD_COPLANAR7): their slow fans may
        # carry B_t through zero, so the ray integrates them in the total
        # pressure, with the answer's own shock masks -- as for planar4
        c7 = np.asarray(res["coplanar7"], dtype=bool)
        m7c = m7[c7[m7]]
        m7 = m7[~c7[m7]]
        if m7c.size:
            sh = [np.asarray(s)[m7c] for s in res["c7_shocks"]]
            kd = [np.asarray(k)[m7c] for k in res["c7_kinds"]]
            from rmhd.batched import planar4_b as _P4B
            st, reg, e = RAY.state_at_xi(
                [c[m7c] for c in subL], [c[m7c] for c in subR],
                [[z[j][m7c] for j in range(7)] for z in res["zones"]],
                [v[m7c] for v in res["VsLv"]], [v[m7c] for v in res["VsRv"]],
                subBn[m7c], gamma, P["xi"], P["fan_p"], P["fan_n"],
                kinds={"LF": sh[0], "LS": sh[1], "RS": sh[2], "RF": sh[3]},
                slow_by_pressure=(kd[0] != _P4B.KIND_B, kd[1] != _P4B.KIND_B))
            for j in range(7):
                star[j][m7c] = st[j]
            region[m7c] = reg
            ray_ok[m7c] = ~e
            n_fan += int(np.isin(reg, RAY.FAN_REGIONS).sum())
    if m7.size:
        st, reg, e = RAY.state_at_xi(
            [c[m7] for c in subL], [c[m7] for c in subR],
            [[z[j][m7] for j in range(7)] for z in res["zones"]],
            [v[m7] for v in res["VsLv"]], [v[m7] for v in res["VsRv"]],
            subBn[m7], gamma, P["xi"], P["fan_p"], P["fan_n"])
        for j in range(7):
            star[j][m7] = st[j]
        region[m7] = reg
        ray_ok[m7] = ~e
        n_fan += int(np.isin(reg, RAY.FAN_REGIONS).sum())

    m3 = np.flatnonzero(conv & ~np.isin(cls, (C.FULL7, C.COPLANAR)))
    if m3.size:
        st, reg, e = RAY.state_at_xi_3wave(
            [c[m3] for c in subL], [c[m3] for c in subR],
            [z[m3] for z in res["zones"][2]], [z[m3] for z in res["zones"][3]],
            res["VsL"][m3], res["VsR"][m3], subBn[m3], gamma, P["xi"],
            P["fan_p"])
        for j in range(7):
            star[j][m3] = st[j]
        region[m3] = reg
        ray_ok[m3] = ~e
        n_fan += int(np.isin(reg, RAY.FAN_REGIONS_3).sum())
    return res, d, star, region, ray_ok, n_fan


def _planar_and_ray(P, gamma, sL5, sR5, B5, *, accuracy, max_iter,
                    unk3_init=None, flip=(False, False)):
    """The five-wave planar solve, the xi = 0 ray and the physical check.

    One lane in, one verdict out, pure numpy: this is the planar rescue's
    body, factored out so that anything else that offers a lane to the planar
    solver -- another seed, another flip branch, the failure ledger -- applies
    EXACTLY production's acceptance and not a private copy of it.

    Returns a dict:
      good     accepted: verified against the full seven-wave residual,
               planar enough, ray resolved, state physical, and (with a flip)
               the fan ordered
      star, region   the state at xi = 0 and its zone (valid where `good`)
      zones6, VsLv, VsRv   the answer in the seven-wave layout
               [A, A2, B, C, D2, D] -- A2 = A and D2 = D without a flip
      n_iter   Newton iterations per lane
      reason   why a lane was refused: 0 accepted, 1 the reduced Newton did
               not converge, 2 the input is not planar, 3 the answer fails
               the full residual, 4 the ray, 5 unphysical, 6 self-crossing

    Without a flip both rotational discontinuities have zero strength and
    cannot cross anything that matters, so the wave-order gate is skipped, as
    it always was; a flipped answer carries a strength-pi rotation and is
    held to the gate.
    """
    import numpy as np
    RAY = P["ray"]
    n = B5.shape[0]
    r5 = P["planar5"]["solve"](sL5, sR5, B5, unk3_init=unk3_init,
                               accuracy=accuracy, max_iter=max_iter,
                               planar_tol=_PLANAR_TOL,
                               verify_tol=_PLANAR5_VERIFY, flip=flip)
    good = r5["converged"].astype(bool)
    raw = r5["converged_raw"].astype(bool)
    reason = np.where(good, 0, np.where(
        ~raw, 1, np.where(r5["planar_resid"] > _PLANAR_TOL, 2, 3))).astype(np.int8)
    A, B_, C_, D = r5["zones"]
    A2, D2 = r5["zones_mid"]
    zones6 = [A, A2, B_, C_, D2, D]
    out = dict(good=good, star=None, region=None, zones6=zones6,
               VsLv=None, VsRv=None, n_iter=np.asarray(r5["n_iter"]),
               reason=reason, unk=r5["unk"],
               # how close each lane came: the reduced system's residual and
               # the full seven-wave one (diagnostics; nothing reads them to
               # decide anything)
               resid=np.asarray(r5["nrm"]),
               full_resid=np.asarray(r5["full_resid"]),
               # how far each slow wave missed the tangential field it was
               # asked to reach: the two equations the reduced system drops
               slack=np.maximum(np.abs(np.asarray(r5["slack"][0])),
                                np.abs(np.asarray(r5["slack"][1]))))
    if not good.any():
        return out
    aspd = P["planar5"]["waves"]["alfven_speed"]
    VsLv5 = [r5["speeds"][0], aspd(A, B5, "L"), r5["speeds"][1]]
    VsRv5 = [r5["speeds"][4], aspd(D, B5, "R"), r5["speeds"][3]]
    st5, reg5, e5 = RAY.state_at_xi(
        sL5, sR5, zones6, VsLv5, VsRv5,
        B5, gamma, P["xi"], P["fan_p"], P["fan_n"])
    reason[good & e5] = 4
    good = good & ~e5
    rho5, P5_, vn5, vt15, vt25, Bt15, Bt25 = st5
    v2_ = vn5 * vn5 + vt15 * vt15 + vt25 * vt25
    W2_ = 1.0 / np.maximum(1.0 - v2_, 1e-300)
    eta_ = B5 * vn5 + Bt15 * vt15 + Bt25 * vt25
    b2_ = (B5 ** 2 + Bt15 ** 2 + Bt25 ** 2) / W2_ + eta_ ** 2
    phys5 = (rho5 > 0.0) & (v2_ < 1.0) & (P5_ - 0.5 * b2_ > 0.0)
    for c_ in st5:
        phys5 &= np.isfinite(c_)
    reason[good & ~phys5] = 5
    good = good & phys5
    if (flip[0] or flip[1]) and _WAVE_ORDER and good.any():
        x5 = _self_crossing(zones6, VsLv5, VsRv5, _WAVE_ORDER_TOL)
        reason[good & x5] = 6
        good = good & ~x5
    if _ADMISSIBLE_GATE and good.any():
        xa = ~_admissible_structure(P, gamma, sL5, sR5, B5, zones6, VsLv5, VsRv5)
        reason[good & xa] = 7                          # refused by the gate
        good = good & ~xa
    out.update(good=good, star=st5, region=reg5, VsLv=VsLv5, VsRv=VsRv5)
    return out


def _planar_seed(sL5, sR5, B5, kind):
    """A planar seed `[ln p_LF, Bt_CD, ln p_RF]` other than the default:
    the mean total pressure with `+-|B_t|` of one side (``"+L"``, ``"-L"``,
    ``"+R"``, ``"-R"``) or `sqrt(2P)` signed like the mean field (``"s2p"``).
    ``None`` is the solver's own default."""
    import numpy as np
    from rmhd.batched import planar5_b as P5B
    if kind is None:
        return None
    L0, R0 = P5B.to_planar(sL5, sR5, B5)[:2]
    d = P5B.default_seed(L0, R0)
    if kind == "s2p":
        bt = np.sqrt(2.0 * np.exp(d[0])) * np.where(d[1] >= 0.0, 1.0, -1.0)
    else:
        bt = np.abs((L0 if kind[1] == "L" else R0)[5])
        bt = bt if kind[0] == "+" else -bt
    return [d[0].copy(), np.array(bt, dtype=float, copy=True), d[0].copy()]


def _planar_ladder(P, gamma, sL5, sR5, B5, *, r5, accuracy, max_iter):
    """Offer the lanes `r5` left unsolved to the rungs of `_PLANAR5_RUNGS`.

    ``r5`` is `_planar_and_ray`'s answer from the default start; the return
    value has the same layout with the rungs' answers merged in, plus
    ``rung`` (0 the default start, k the k-th rung, -1 unsolved) and
    ``n_iter`` summed over the attempts.  Every answer passes through
    `_planar_and_ray`, that is through production's acceptance.
    """
    import numpy as np
    n = B5.shape[0]
    good = r5["good"].copy()
    rung = np.where(good, 0, -1).astype(np.int8)
    n_iter = np.maximum(np.asarray(r5["n_iter"]), 0).astype(int)

    def full(v):
        return (np.zeros(n) if v is None
                else np.array(np.broadcast_to(v, (n,)), dtype=float, copy=True))

    have = r5["star"] is not None
    star = [full(r5["star"][j] if have else None) for j in range(7)]
    region = (np.array(r5["region"], copy=True) if have
              else np.zeros(n, dtype=int))
    zones6 = [[full(c) for c in z] for z in r5["zones6"]]
    VsLv = [full(r5["VsLv"][j] if have else None) for j in range(3)]
    VsRv = [full(r5["VsRv"][j] if have else None) for j in range(3)]
    reason = r5["reason"].copy()
    for k, (fl, fr, kind) in enumerate(_PLANAR5_RUNGS, start=1):
        i = np.flatnonzero(~good)
        if i.size == 0:
            break
        sl, sr, b = [c[i] for c in sL5], [c[i] for c in sR5], B5[i]
        r = _planar_and_ray(P, gamma, sl, sr, b, accuracy=accuracy,
                            max_iter=max_iter, flip=(fl, fr),
                            unk3_init=_planar_seed(sl, sr, b, kind))
        n_iter[i] += np.maximum(np.asarray(r["n_iter"]), 0)
        g = r["good"]
        if not g.any():
            continue
        w = i[g]
        for j in range(7):
            star[j][w] = r["star"][j][g]
        region[w] = np.asarray(r["region"])[g]
        for z, zr in zip(zones6, r["zones6"]):
            for j in range(7):
                z[j][w] = np.broadcast_to(zr[j], (i.size,))[g]
        for j in range(3):
            VsLv[j][w] = np.broadcast_to(r["VsLv"][j], (i.size,))[g]
            VsRv[j][w] = np.broadcast_to(r["VsRv"][j], (i.size,))[g]
        good[w] = True
        rung[w] = k
        reason[w] = 0
    out = dict(r5)
    out.update(good=good, star=star, region=region, zones6=zones6,
               VsLv=VsLv, VsRv=VsRv, n_iter=n_iter, reason=reason, rung=rung)
    return out


def _coplanar7_and_ray(P, gamma, sL7, sR7, B7, *, seeds, accuracy):
    """The planar seven-wave rung (``RMHD_COPLANAR7=late``): coplanar7_b's
    solve from the sweep's network seeds, the xi = 0 ray with the answer's own
    shock masks and slow-wave steering (as `_solve_and_ray` resolves the same
    answers when they come out of solve_batch), and production's acceptance:
    a physical state, the wave order, the admissibility gate.

    ``seeds``: ``seed6`` (the primary checkpoint's clamped seed, six per-lane
    arrays) and ``extra`` (one such seed per extra checkpoint, or None).
    Returns `_planar_and_ray`'s layout plus ``stage`` (1-4, 0 unanswered) and
    ``rot`` (two per-lane masks); ``reason`` 0 accepted, 1 no branch
    converged, 4 the ray, 5 unphysical, 6 waves out of order, 7 refused by
    the gate.
    """
    import numpy as np
    from rmhd.batched import planar4_b as P4B
    RAY = P["ray"]
    n = B7.shape[0]
    r7 = P["coplanar7"]["solve"](sL7, sR7, B7, seeds["seed6"],
                                 seeds_extra=seeds.get("extra"),
                                 accuracy=accuracy)
    good = np.asarray(r7["converged"], dtype=bool)
    reason = np.where(good, 0, 1).astype(np.int8)
    zones6, VsLv, VsRv = r7["zones"], r7["VsLv"], r7["VsRv"]
    out = dict(good=good, star=None, region=None, zones6=zones6,
               VsLv=VsLv, VsRv=VsRv, n_iter=np.asarray(r7["n_iter"]),
               reason=reason, stage=np.asarray(r7["stage"]),
               rot=[np.asarray(x, dtype=bool) for x in r7["rot"]],
               steer=[np.asarray(k, dtype=np.int8) for k in r7["kinds"]])
    if not good.any():
        return out
    i = np.flatnonzero(good)
    sh = [np.asarray(s)[i] for s in r7["shocks"]]
    kd = [np.asarray(k)[i] for k in r7["kinds"]]
    st, reg, e = RAY.state_at_xi(
        [c[i] for c in sL7], [c[i] for c in sR7],
        [[z[j][i] for j in range(7)] for z in zones6],
        [v[i] for v in VsLv], [v[i] for v in VsRv],
        B7[i], gamma, P["xi"], P["fan_p"], P["fan_n"],
        kinds={"LF": sh[0], "LS": sh[1], "RS": sh[2], "RF": sh[3]},
        slow_by_pressure=(kd[0] != P4B.KIND_B, kd[1] != P4B.KIND_B))
    star = [np.zeros(n) for _ in range(7)]
    region = np.full(n, -1, dtype=int)
    for j in range(7):
        star[j][i] = st[j]
    region[i] = reg
    ray_bad = np.zeros(n, dtype=bool)
    ray_bad[i] = e
    reason[good & ray_bad] = 4
    good = good & ~ray_bad
    rho7, P7_, vn7, vt17, vt27, Bt17, Bt27 = star
    v2_ = vn7 * vn7 + vt17 * vt17 + vt27 * vt27
    W2_ = 1.0 / np.maximum(1.0 - v2_, 1e-300)
    eta_ = B7 * vn7 + Bt17 * vt17 + Bt27 * vt27
    b2_ = (B7 ** 2 + Bt17 ** 2 + Bt27 ** 2) / W2_ + eta_ ** 2
    phys7 = (rho7 > 0.0) & (v2_ < 1.0) & (P7_ - 0.5 * b2_ > 0.0)
    for c_ in star:
        phys7 &= np.isfinite(c_)
    reason[good & ~phys7] = 5
    good = good & phys7
    if _WAVE_ORDER and good.any():
        x7 = _self_crossing(zones6, VsLv, VsRv, _WAVE_ORDER_TOL)
        reason[good & x7] = 6
        good = good & ~x7
    if _ADMISSIBLE_GATE and good.any():
        xa = ~_admissible_structure(P, gamma, sL7, sR7, B7, zones6, VsLv, VsRv)
        reason[good & xa] = 7                          # refused by the gate
        good = good & ~xa
    out.update(good=good, star=star, region=region)
    return out


def _continuation_and_ray(P, gamma, sL, sR, Bn, **kw):
    """The continuation rung's body (src/physics/continuation.py), named
    here so that `exact_pool.map_lanes` can run it on the workers."""
    from . import continuation as CT
    return CT.continuation_and_ray(P, gamma, sL, sR, Bn, **kw)


def _ml_seeds(model, scaler, L, R, Bn, n_retries):
    """Production's seeding for arbitrary solver-frame states: the primary
    checkpoint's candidate 0 (clamped), per-lane retry keys, and the ensemble
    -- the primary's further candidates, then one seed per extra checkpoint.
    The same calls in the same order as the sweep's own seeding."""
    import numpy as np
    from rmhd.batched import ml_b as MB
    UL, UR = np.stack(L, axis=1), np.stack(R, axis=1)
    cands, feats = MB.predict_unk6_k(model, scaler, UL, UR, Bn)
    seed6, _ = MB.clamp_seed_physical(cands[0], L, R, Bn)
    keys = MB.lane_keys(feats) if n_retries else None
    extra = None
    if n_retries and (len(cands) > 1 or _EXTRA_CKPTS):
        extra = []
        for u_k in cands[1:]:
            extra.append(MB.clamp_seed_physical(u_k, L, R, Bn)[0])
        for ck in _EXTRA_CKPTS:
            m_k, s_k = _load_model_cached(_rmhd_paths.resolve(ck))
            u_k, _ = MB.predict_unk6(m_k, s_k, UL, UR, Bn)
            extra.append(MB.clamp_seed_physical(u_k, L, R, Bn)[0])
    return dict(seed6=seed6, keys=keys, extra=extra)


def _planar4_accept(P, gamma, r, sL4, sR4, B4):
    """Production's acceptance of a planar4 ``solve`` answer beyond its own
    certificate: the xi = 0 ray resolved and the state there physical.
    Returns ``(good, reason, star, region, zones6, VsLv, VsRv)``; the last four
    are None when no lane converged."""
    import numpy as np
    from rmhd.batched import planar4_b as P4B
    RAY = P["ray"]
    S4 = P["planar4"]
    good = r["converged"].astype(bool)
    reason = r["reason"].astype(np.int8).copy()
    A, B_, C_, D = r["zones"]
    zones6 = [A, A, B_, C_, D, D]
    if "zones_rot" in r:                       # a rotation answer (RMHD_PLANAR4_ROT)
        zones6 = [A, r["zones_rot"][0], B_, C_, r["zones_rot"][1], D]
    if not good.any():
        return good, reason, None, None, zones6, None, None
    VsLv = [r["speeds"][0], S4["alfven_speed"](A, B4, "L"), r["speeds"][1]]
    VsRv = [r["speeds"][4], S4["alfven_speed"](D, B4, "R"), r["speeds"][3]]
    sh = r["shocks"]
    st, reg, e = RAY.state_at_xi(
        sL4, sR4, zones6, VsLv, VsRv, B4, gamma, P["xi"], P["fan_p"],
        P["fan_n"], kinds={"LF": sh[0], "LS": sh[1], "RS": sh[2], "RF": sh[3]},
        slow_by_pressure=(r["kinds"][0] != P4B.KIND_B,
                          r["kinds"][1] != P4B.KIND_B))
    reason[good & e] = 6
    good = good & ~e
    rho, Pt, vn, vt1, vt2, Bt1, Bt2 = st
    v2_ = vn * vn + vt1 * vt1 + vt2 * vt2
    W2_ = 1.0 / np.maximum(1.0 - v2_, 1e-300)
    eta_ = B4 * vn + Bt1 * vt1 + Bt2 * vt2
    b2_ = (B4 ** 2 + Bt1 ** 2 + Bt2 ** 2) / W2_ + eta_ ** 2
    phys = (rho > 0.0) & (v2_ < 1.0) & (Pt - 0.5 * b2_ > 0.0)
    for c_ in st:
        phys &= np.isfinite(c_)
    reason[good & ~phys] = 7
    good = good & phys
    if _ADMISSIBLE_GATE and good.any():
        xa = ~_admissible_structure(P, gamma, sL4, sR4, B4, zones6, VsLv, VsRv)
        reason[good & xa] = 9                          # refused by the gate
        good = good & ~xa
    return good, reason, st, reg, zones6, VsLv, VsRv


def _planar4_and_ray(P, gamma, sL4, sR4, B4, *, accuracy, max_iter, hint=None):
    """The four-unknown planar solve, the xi = 0 ray and the physical check
    -- the rung's body, so that the failure ledger can offer lanes to it
    under exactly production's acceptance.

    ``hint`` (with RMHD_PLANAR4_STAR) is ``{"p": HLLD's star pressure}`` per
    lane, the start of the second attempt (`_planar4_star`).

    Returns a dict: ``good``, ``star``, ``region``, ``zones6``
    (``[A, A, B, C, D, D]``), ``VsLv``, ``VsRv``, ``n_iter``, ``kinds`` (2, N),
    ``reason`` (0 accepted, 1 not converged, 2 not planar, 3 fails the
    certificate, 4 fan not ordered, 5 cannot start, 6 ray, 7 unphysical),
    ``cert_resid``, ``full_resid``.
    """
    import numpy as np
    S4 = P["planar4"]
    r = S4["solve"](sL4, sR4, B4, accuracy=accuracy, max_iter=max_iter,
                    planar_tol=_PLANAR_TOL, verify_tol=_PLANAR5_VERIFY)
    good, reason, st, reg, zones6, VsLv, VsRv = _planar4_accept(
        P, gamma, r, sL4, sR4, B4)
    out = dict(good=good, star=st, region=reg, zones6=zones6,
               VsLv=VsLv, VsRv=VsRv, n_iter=np.asarray(r["n_iter"]),
               kinds=r["kinds"], reason=reason, unk=r["unk"],
               cert_resid=np.asarray(r["cert_resid"]),
               full_resid=np.asarray(r["full_resid"]),
               resid=np.asarray(r["nrm"]))
    if _PLANAR4_ROT:
        out["refused_seed"] = [np.full(B4.shape[0], np.nan) for _ in range(4)]
        _keep_refused(out, np.arange(B4.shape[0]), r, reason == 9)
    if _PLANAR4_STAR and hint is not None and not good.all():
        star = _planar4_star_batched if _PLANAR4_STAR_BATCH else _planar4_star
        out = star(P, gamma, sL4, sR4, B4, out, hint["p"],
                   accuracy=accuracy, max_iter=max_iter)
    if _PLANAR4_ROT and not out["good"].all():
        out = _planar4_rotation(P, gamma, sL4, sR4, B4, out,
                                accuracy=accuracy, max_iter=max_iter)
    return out


def _keep_refused(out, i, r, refused):
    """Remember, as the rotation re-solve's start (`_planar4_rotation`), the
    first answer of lanes ``i[refused]`` that converged but was refused: its
    four unknowns in the signed-field form [ln P behind LF, B_t behind LS, B_t
    behind RS, ln P behind RF], planar frame."""
    import numpy as np
    seed = out["refused_seed"]
    k = np.flatnonzero(np.asarray(refused, dtype=bool) & np.isnan(seed[0][i]))
    if k.size == 0:
        return
    ca, sa = np.cos(r["alpha"][k]), np.sin(r["alpha"][k])
    B_, C_ = r["zones"][1], r["zones"][2]
    w = i[k]
    seed[0][w] = r["unk"][0][k]
    seed[1][w] = ca * B_[5][k] - sa * B_[6][k]
    seed[2][w] = ca * C_[5][k] - sa * C_[6][k]
    seed[3][w] = r["unk"][3][k]


def _harvest_rescue(harvester, C, subL, subR, subBn, w, good, r, *, source,
                    steer, rot, stage):
    """Record a rescue rung's accepted answers as harvest rows
    (RMHD_HARVEST_RESCUES): ``source`` 2 the planar seven-wave rung, 3 the
    four-unknown solver, with each row's slow-wave steering and rotations
    (2, k) and stage (k).  ``w`` are the accepted lanes in the sweep's
    selection, ``good`` the same lanes as a mask over the rung's candidates,
    ``r`` the rung's result (its ``zones6``, ``VsLv``, ``VsRv`` over the
    candidates).  Restricted to the accepted lanes, as the five-wave rows
    are: `record`'s unsolved branch fires on every call."""
    import numpy as np
    k = int(w.size)
    harvester.record(
        [c[w] for c in subL], [c[w] for c in subR], subBn[w],
        cls=np.full(k, C.COPLANAR, dtype=np.int8),
        converged=np.ones(k, dtype=bool),
        zones=[[np.asarray(c)[good] for c in z] for z in r["zones6"]],
        VsLv=[np.asarray(v)[good] for v in r["VsLv"]],
        VsRv=[np.asarray(v)[good] for v in r["VsRv"]],
        attempts=np.full(k, 2, dtype=int),
        accepted=np.ones(k, dtype=bool),
        seven_wave=np.ones(k, dtype=bool),
        source=source,
        extras=dict(steer=steer, rot=rot, stage=stage))


def _planar4_take(out, w, g, r, st, reg, zones6, VsLv, VsRv, n):
    """Write the accepted lanes ``g`` of a four-unknown answer ``r`` (solved on
    the sub-batch whose lanes are ``w`` of ``out``) into ``out``."""
    import numpy as np
    if out["star"] is None:
        out["star"] = [np.zeros(n) for _ in range(7)]
        out["region"] = np.zeros(n, dtype=np.asarray(reg).dtype)
        out["VsLv"] = [np.zeros(n) for _ in range(3)]
        out["VsRv"] = [np.zeros(n) for _ in range(3)]
    for j in range(7):
        out["star"][j][w] = st[j][g]
    out["region"][w] = np.asarray(reg)[g]
    for k in range(3):
        out["VsLv"][k][w] = VsLv[k][g]
        out["VsRv"][k][w] = VsRv[k][g]
    for z, z2 in zip(out["zones6"], zones6):
        for j in range(7):
            z[j][w] = z2[j][g]
    out["kinds"][:, w] = r["kinds"][:, g]
    out["reason"][w] = 0
    for c in range(4):
        out["unk"][c][w] = r["unk"][c][g]
    for k, k2 in (("cert_resid", "cert_resid"), ("full_resid", "full_resid"),
                  ("resid", "nrm")):
        out[k] = np.asarray(out[k]).copy()
        out[k][w] = np.asarray(r[k2])[g]
    out["good"] = out["good"].copy()
    out["good"][w] = True


def _planar4_rotation(P, gamma, sL4, sR4, B4, out, *, accuracy, max_iter):
    """The lanes still lost that had a refused four-unknown answer, solved
    again from it with the coplanar rotation in front of the slow waves (see
    _PLANAR4_ROT); the first combination whose answer production accepts and
    `_admissible_structure` passes is kept.  ``out`` is updated in place and
    returned."""
    import numpy as np
    S4 = P["planar4"]
    n = B4.shape[0]
    seed = out["refused_seed"]
    sub = lambda X, i: [c[i] for c in X]
    # zones6 is [A, A, B, C, D, D] with A and D shared: a rotation answer's R3
    # and R6 differ from R2 and R7, so give every slot its own arrays first
    out["zones6"] = [[np.array(c, dtype=float, copy=True) for c in z] for z in out["zones6"]]
    if _PLANAR4_STAR_BATCH:
        # all three choices of every lane in one solve, per lane the first
        # accepted in _PLANAR4_ROT_COMBOS order (the serial loop's answer)
        i0 = np.flatnonzero(~out["good"] & np.isfinite(seed[0]))
        if i0.size == 0:
            return out
        nc = len(_PLANAR4_ROT_COMBOS)
        i = np.tile(i0, nc)
        cidx = np.repeat(np.arange(nc), i0.size)
        rot = np.array(_PLANAR4_ROT_COMBOS, dtype=bool)[cidx].T
        kinds = np.full((2, i.size), 2, dtype=np.int8)
        r = S4["solve"](sub(sL4, i), sub(sR4, i), B4[i], accuracy=accuracy,
                        max_iter=min(max_iter, _PLANAR4_STAR_ITER),
                        planar_tol=_PLANAR_TOL, verify_tol=_PLANAR5_VERIFY,
                        kinds=kinds, retry=False, unk_init=[c[i].copy() for c in seed],
                        central=True, rot=rot)
        good, reason, st, reg, zones6, VsLv, VsRv = _planar4_accept(
            P, gamma, r, sub(sL4, i), sub(sR4, i), B4[i])
        if good.any() and not _ADMISSIBLE_GATE:
            good &= _admissible_structure(P, gamma, sub(sL4, i), sub(sR4, i), B4[i],
                                          zones6, VsLv, VsRv)
        out["n_iter"] = np.asarray(out["n_iter"]).copy()
        np.add.at(out["n_iter"], i, np.asarray(r["n_iter"]))
        gi = np.flatnonzero(good)          # branches are combination-major
        if gi.size:
            _, first = np.unique(i[gi], return_index=True)
            g = gi[first]
            _planar4_take(out, i[g], g, r, st, reg, zones6, VsLv, VsRv, n)
        return out
    for combo in _PLANAR4_ROT_COMBOS:
        i = np.flatnonzero(~out["good"] & np.isfinite(seed[0]))
        if i.size == 0:
            break
        kinds = np.full((2, i.size), 2, dtype=np.int8)       # signed-field slow waves
        rot = np.tile(np.array(combo, dtype=bool)[:, None], (1, i.size))
        r = S4["solve"](sub(sL4, i), sub(sR4, i), B4[i], accuracy=accuracy,
                        max_iter=min(max_iter, _PLANAR4_STAR_ITER),
                        planar_tol=_PLANAR_TOL, verify_tol=_PLANAR5_VERIFY,
                        kinds=kinds, retry=False, unk_init=[c[i].copy() for c in seed],
                        central=True, rot=rot)
        good, reason, st, reg, zones6, VsLv, VsRv = _planar4_accept(
            P, gamma, r, sub(sL4, i), sub(sR4, i), B4[i])
        if good.any() and not _ADMISSIBLE_GATE:
            good &= _admissible_structure(P, gamma, sub(sL4, i), sub(sR4, i), B4[i],
                                          zones6, VsLv, VsRv)
        out["n_iter"] = np.asarray(out["n_iter"]).copy()
        out["n_iter"][i] += np.asarray(r["n_iter"])
        if good.any():
            _planar4_take(out, i[good], np.flatnonzero(good), r, st, reg, zones6,
                          VsLv, VsRv, n)
    return out


def _shock_entropy_ratio(r, sL4, sR4, B4, gamma):
    """The smallest K_behind / K_ahead (K = p / rho^gamma) over the shocks of
    a planar4 answer; 1 where it has none."""
    import numpy as np

    def K(X):
        rho, Pt, vx, vy, vz, By, Bz = X
        v2 = vx * vx + vy * vy + vz * vz
        W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
        vB = vx * B4 + vy * By + vz * Bz
        return (Pt - 0.5 * ((B4 ** 2 + By ** 2 + Bz ** 2) / W2 + vB ** 2)) / X[0] ** gamma

    A, B_, C_, D = r["zones"]
    m = np.ones(B4.shape[0])
    with np.errstate(all="ignore"):
        for row, (ah, bh) in enumerate(((sL4, A), (A, B_), (D, C_), (sR4, D))):
            ratio = np.where(np.isfinite(K(bh) / K(ah)), K(bh) / K(ah), 0.0)
            m = np.where(r["shocks"][row], np.minimum(m, ratio), m)
    return m


def _planar4_star(P, gamma, sL4, sR4, B4, out, p_hint, *, accuracy, max_iter):
    """The lanes `_planar4_and_ray` lost, again from HLLD's star pressure
    (see _PLANAR4_STAR): every start of _PLANAR4_STAR_STARTS, then of
    _PLANAR4_STAR_SIDE, on the lanes still lost, the first accepted answer
    kept.  ``out`` is updated in place on the lanes it answers and returned."""
    import numpy as np
    S4 = P["planar4"]
    W = S4["waves"]
    n = B4.shape[0]
    L0, R0 = S4["to_planar"](sL4, sR4, B4)[:2]
    lp = np.log(np.maximum(np.asarray(p_hint, dtype=float), 1e-12))
    sub = lambda X, i: [c[i] for c in X]
    starts = ([st + (None, 0.0) for st in _PLANAR4_STAR_STARTS]
              + list(_PLANAR4_STAR_SIDE))
    for central, kd, dl, fast, off in starts:
        i = np.flatnonzero(~out["good"])
        if i.size == 0:
            break
        if fast == "near":
            # only a lane with a weak fast wave has something new to try
            i = i[(np.abs(np.exp(lp[i]) / L0[1][i] - 1.0) < 0.02)
                  | (np.abs(np.exp(lp[i]) / R0[1][i] - 1.0) < 0.02)]
            if i.size == 0:
                continue
        pl, pr = np.exp(lp[i]), np.exp(lp[i])
        if fast == "near":
            for pp, P0 in ((pl, L0[1][i]), (pr, R0[1][i])):
                near = np.abs(pp / P0 - 1.0) < 0.02
                pp[near] = P0[near] * (1.0 + off)
        elif fast == "both":
            pl, pr = L0[1][i] * (1.0 + off), R0[1][i] * (1.0 + off)
        with np.errstate(all="ignore"):
            A, _, eA = W["fast_wave"](pl, sub(L0, i), "LF", B4[i])
            D, _, eD = W["fast_wave"](pr, sub(R0, i), "RF", B4[i])
            sl = (np.log(np.maximum(A[1] * (1.0 - dl), 1e-300)) if kd[0] == 1
                  else A[5] * (1.0 + dl))
            sr = (np.log(np.maximum(D[1] * (1.0 - dl), 1e-300)) if kd[1] == 1
                  else D[5] * (1.0 + dl))
        kinds = np.tile(np.array(kd, dtype=np.int8)[:, None], (1, i.size))
        r = S4["solve"](sub(sL4, i), sub(sR4, i), B4[i], accuracy=accuracy,
                        max_iter=min(max_iter, _PLANAR4_STAR_ITER),
                        planar_tol=_PLANAR_TOL,
                        verify_tol=_PLANAR5_VERIFY, kinds=kinds, retry=False,
                        unk_init=[np.log(pl), sl, sr, np.log(pr)], central=central)
        good, reason, st, reg, zones6, VsLv, VsRv = _planar4_accept(
            P, gamma, r, sub(sL4, i), sub(sR4, i), B4[i])
        good = good & ~(eA | eD)
        pre = good.copy()
        if good.any():
            good &= _shock_entropy_ratio(r, sub(sL4, i), sub(sR4, i), B4[i],
                                         gamma) >= 1.0 - _PLANAR4_STAR_DS
        if "refused_seed" in out:
            _keep_refused(out, i, r, ((reason == 9) & ~(eA | eD)) | (pre & ~good))
        out["n_iter"] = np.asarray(out["n_iter"]).copy()
        out["n_iter"][i] += np.asarray(r["n_iter"])
        if not good.any():
            continue
        w, g = i[good], np.flatnonzero(good)
        if out["star"] is None:
            out["star"] = [np.zeros(n) for _ in range(7)]
            out["region"] = np.zeros(n, dtype=np.asarray(reg).dtype)
            out["VsLv"] = [np.zeros(n) for _ in range(3)]
            out["VsRv"] = [np.zeros(n) for _ in range(3)]
        for j in range(7):
            out["star"][j][w] = st[j][g]
        out["region"][w] = np.asarray(reg)[g]
        for k in range(3):
            out["VsLv"][k][w] = VsLv[k][g]
            out["VsRv"][k][w] = VsRv[k][g]
        for z, z2 in zip(out["zones6"], zones6):
            for j in range(7):
                z[j][w] = z2[j][g]
        out["kinds"][:, w] = r["kinds"][:, g]
        out["reason"][w] = 0
        for c in range(4):
            out["unk"][c][w] = r["unk"][c][g]
        for k, k2 in (("cert_resid", "cert_resid"), ("full_resid", "full_resid"),
                      ("resid", "nrm")):
            out[k] = np.asarray(out[k]).copy()
            out[k][w] = np.asarray(r[k2])[g]
        out["good"] = out["good"].copy()
        out["good"][w] = True
    return out


def _planar4_star_batched(P, gamma, sL4, sR4, B4, out, p_hint, *, accuracy, max_iter):
    """`_planar4_star` with every start of every lane in ONE batch per
    difference scheme (forward, central) instead of one solve per start.

    Each lane's Newton does not see the others (the solves are batch-
    independent), so taking per lane the first accepted start in the original
    order gives the serial loop's answers; what changes is the latency: a lane
    that exhausts the starts waits for two solves instead of eight, and a run's
    sweep waits for its slowest lane (RMHD_PLANAR4_STAR_BATCH=1).  Lanes the
    serial loop would have answered early also run their later starts -- more
    CPU, on cores the runs leave idle."""
    import numpy as np
    S4 = P["planar4"]
    W = S4["waves"]
    n = B4.shape[0]
    L0, R0 = S4["to_planar"](sL4, sR4, B4)[:2]
    lp = np.log(np.maximum(np.asarray(p_hint, dtype=float), 1e-12))
    sub = lambda X, i: [c[i] for c in X]
    starts = ([st + (None, 0.0) for st in _PLANAR4_STAR_STARTS]
              + list(_PLANAR4_STAR_SIDE))
    todo = np.flatnonzero(~out["good"])
    if todo.size == 0:
        return out
    # every (lane, start) branch and its start
    lanes, sidx, U, K, ok_fast = [], [], [[], [], [], []], [], []
    for s, (central, kd, dl, fast, off) in enumerate(starts):
        i = todo
        if fast == "near":
            i = i[(np.abs(np.exp(lp[i]) / L0[1][i] - 1.0) < 0.02)
                  | (np.abs(np.exp(lp[i]) / R0[1][i] - 1.0) < 0.02)]
            if i.size == 0:
                continue
        pl, pr = np.exp(lp[i]), np.exp(lp[i])
        if fast == "near":
            for pp, P0 in ((pl, L0[1][i]), (pr, R0[1][i])):
                near = np.abs(pp / P0 - 1.0) < 0.02
                pp[near] = P0[near] * (1.0 + off)
        elif fast == "both":
            pl, pr = L0[1][i] * (1.0 + off), R0[1][i] * (1.0 + off)
        with np.errstate(all="ignore"):
            A, _, eA = W["fast_wave"](pl, sub(L0, i), "LF", B4[i])
            D, _, eD = W["fast_wave"](pr, sub(R0, i), "RF", B4[i])
            sl = (np.log(np.maximum(A[1] * (1.0 - dl), 1e-300)) if kd[0] == 1
                  else A[5] * (1.0 + dl))
            sr = (np.log(np.maximum(D[1] * (1.0 - dl), 1e-300)) if kd[1] == 1
                  else D[5] * (1.0 + dl))
        lanes.append(i)
        sidx.append(np.full(i.size, s))
        for c, v in enumerate((np.log(pl), sl, sr, np.log(pr))):
            U[c].append(v)
        K.append(np.tile(np.array(kd, dtype=np.int8)[:, None], (1, i.size)))
        ok_fast.append(~(eA | eD))
    lanes = np.concatenate(lanes)
    sidx = np.concatenate(sidx)
    U = [np.concatenate(u) for u in U]
    K = np.concatenate(K, axis=1)
    ok_fast = np.concatenate(ok_fast)
    cen = np.array([bool(starts[s][0]) for s in sidx])
    # one solve per difference scheme, accepted exactly as the serial loop does
    good_b = np.zeros(lanes.size, dtype=bool)
    refused_b = np.zeros(lanes.size, dtype=bool)
    res = {}
    for flag in (False, True):
        b = np.flatnonzero(cen == flag)
        if b.size == 0:
            continue
        li = lanes[b]
        r = S4["solve"](sub(sL4, li), sub(sR4, li), B4[li], accuracy=accuracy,
                        max_iter=min(max_iter, _PLANAR4_STAR_ITER),
                        planar_tol=_PLANAR_TOL, verify_tol=_PLANAR5_VERIFY,
                        kinds=K[:, b], retry=False, unk_init=[u[b] for u in U],
                        central=flag)
        good, reason, st, reg, zones6, VsLv, VsRv = _planar4_accept(
            P, gamma, r, sub(sL4, li), sub(sR4, li), B4[li])
        good = good & ok_fast[b]
        pre = good.copy()
        if good.any():
            good &= _shock_entropy_ratio(r, sub(sL4, li), sub(sR4, li), B4[li],
                                         gamma) >= 1.0 - _PLANAR4_STAR_DS
        good_b[b] = good
        refused_b[b] = ((reason == 9) & ok_fast[b]) | (pre & ~good)
        res[flag] = (b, r, st, reg, zones6, VsLv, VsRv)
        out["n_iter"] = np.asarray(out["n_iter"]).copy()
        np.add.at(out["n_iter"], li, np.asarray(r["n_iter"]))
    # per lane, the first accepted start in the serial order
    order = np.lexsort((sidx, lanes))
    lo, go = lanes[order], good_b[order]
    if "refused_seed" in out:
        # the serial loop keeps the first refused answer in start order: the
        # forward start comes first, and within a scheme the branches are in
        # start order, so per lane the first refused branch of each scheme,
        # forward scheme first (_keep_refused never overwrites a kept seed)
        for flag in (False, True):
            if flag not in res:
                continue
            b, r = res[flag][0], res[flag][1]
            idx = np.flatnonzero(refused_b[b])
            if idx.size == 0:
                continue
            _, f = np.unique(lanes[b][idx], return_index=True)
            mask = np.zeros(b.size, dtype=bool)
            mask[idx[f]] = True
            _keep_refused(out, lanes[b], r, mask)
    gi = np.flatnonzero(go)
    if gi.size == 0:
        return out
    _, first = np.unique(lo[gi], return_index=True)
    chosen = order[gi[first]]                     # branch index per answered lane
    for flag, (b, r, st, reg, zones6, VsLv, VsRv) in res.items():
        pos = np.searchsorted(b, chosen)
        inb = (pos < b.size) & (b[np.minimum(pos, b.size - 1)] == chosen)
        if not inb.any():
            continue
        g = pos[inb]
        w = lanes[chosen[inb]]
        if out["star"] is None:
            out["star"] = [np.zeros(n) for _ in range(7)]
            out["region"] = np.zeros(n, dtype=np.asarray(reg).dtype)
            out["VsLv"] = [np.zeros(n) for _ in range(3)]
            out["VsRv"] = [np.zeros(n) for _ in range(3)]
        for j in range(7):
            out["star"][j][w] = st[j][g]
        out["region"][w] = np.asarray(reg)[g]
        for k in range(3):
            out["VsLv"][k][w] = VsLv[k][g]
            out["VsRv"][k][w] = VsRv[k][g]
        for z, z2 in zip(out["zones6"], zones6):
            for j in range(7):
                z[j][w] = z2[j][g]
        out["kinds"][:, w] = r["kinds"][:, g]
        out["reason"][w] = 0
        for c in range(4):
            out["unk"][c][w] = r["unk"][c][g]
        for k, k2 in (("cert_resid", "cert_resid"), ("full_resid", "full_resid"),
                      ("resid", "nrm")):
            out[k] = np.asarray(out[k]).copy()
            out[k][w] = np.asarray(r[k2])[g]
        out["good"] = out["good"].copy()
        out["good"][w] = True
    return out


def _run_pipeline(P, gamma, subL, subR, subBn, seed6, keys, seeds_extra, **kw):
    """In-process, or over the worker pool when RMHD_POOL asks for one."""
    from . import exact_pool
    cfg = exact_pool.config()
    if cfg is None or int(subBn.shape[0]) < cfg["min_lanes"]:
        return _solve_and_ray(P, gamma, subL, subR, subBn, seed6, keys,
                              seeds_extra, **kw)
    return exact_pool.solve_and_ray(gamma, subL, subR, subBn, seed6, keys,
                                    seeds_extra, **kw)


def exact_flux_batched(sL, sR, eos, idir: int = 0, *, model=None, scaler=None,
                       fallback=hlld_flux, tau_weak: float = 1e-6,
                       tau_bt: float = 1e-9,
                       accuracy: float = 1e-8, max_iter: int = 40,
                       n_retries: int = 0, harvester=None,
                       _record_idir=None, _rendezvous=None):
    """Godunov flux from the exact solution at ``xi = 0``, batched.

    Same contract as :func:`exact_flux` and as ``hlld_flux``, so ``sweep()``
    needs only the name.  Every interface of a sweep is solved in one call.

    The structure of the routine is: HLLD everywhere first, then overwrite the
    lanes the exact solver actually owns.  That ordering is deliberate -- a
    lane that fails ANY gate keeps a valid flux rather than a hole, and the
    negative ``p_star`` sentinel ``hlld_flux`` wrote survives on exactly those
    lanes, which is what ``tests/test_idir.py``'s branch-flip logic reads.

    With ``RMHD_CANONICAL`` every face is solved as the canonical member of its
    symmetry orbit in the x-frame and the answer mapped back
    (``src/physics/canonical.py``); ``_record_idir`` is that call's way of
    passing the face direction on to the harvest's coverage map.

    ``_rendezvous`` (``RMHD_MERGE_SWEEPS``, `sweep_merge`): the per-lane
    pipeline call is handed to it, which runs it once for the sweeps of an RK
    stage; everything else here runs per sweep, in the sweeps' order.
    """
    if _CANONICAL and _record_idir is None:
        from .canonical import canonical_flux
        return canonical_flux(
            exact_flux_batched, sL, sR, eos, idir, model=model, scaler=scaler,
            fallback=fallback, tau_weak=tau_weak, tau_bt=tau_bt,
            accuracy=accuracy, max_iter=max_iter, n_retries=n_retries,
            harvester=harvester, _rendezvous=_rendezvous)
    rec_idir = idir if _record_idir is None else _record_idir
    P = _batched_parts(_check_eos(eos))
    np = P["np"]
    RAY = P["ray"]
    gamma = _check_eos(eos)
    N = sL["rho"].numel()
    dt = sL["rho"].dtype

    F, U, p_star = fallback(sL, sR, eos, idir=idir)
    F = {k: v.clone() for k, v in F.items()}
    U = {k: v.clone() for k, v in U.items()}
    p_star = p_star.clone()
    # HLLD's star pressure, before any exact answer overwrites it: the
    # four-unknown rung's second start (_PLANAR4_STAR)
    p_hll = (p_star.detach().abs().reshape(-1).cpu().numpy().astype(float)
             if _PLANAR4_STAR else None)

    (L7, BnL), (R7, BnR) = to_solver_frame(sL, eos, idir), to_solver_frame(sR, eos, idir)
    dBn = float((BnL - BnR).abs().max())
    if dBn > 1e-10 * float(BnL.abs().max().clamp(min=1e-30)):
        raise ValueError(f"normal field is not single-valued at the face "
                         f"(max |Bn_L - Bn_R| = {dBn:.3e})")

    to_np = lambda t: t.detach().cpu().numpy().astype(float)
    left = [to_np(c) for c in L7]
    right = [to_np(c) for c in R7]
    Bn = to_np(BnL)

    jump = relative_jump(sL, sR)
    v2L = sL["vx"] ** 2 + sL["vy"] ** 2 + sL["vz"] ** 2
    v2R = sR["vx"] ** 2 + sR["vy"] ** 2 + sR["vz"] ** 2
    bad = (v2L >= 1.0) | (v2R >= 1.0) | (sL["rho"] <= 0) | (sR["rho"] <= 0)
    weak = (~bad) & (jump < tau_weak)
    n_weak_linear = 0
    weak_linear = torch.zeros(N, dtype=torch.bool)
    if _WEAK_FLUX == "linear" and bool(weak.any()):
        from .linearised import linearised_flux
        wi = torch.nonzero(weak.reshape(-1)).reshape(-1)
        Fw, Uw, pw, okw = linearised_flux(
            {k: v.reshape(-1)[wi] for k, v in sL.items()},
            {k: v.reshape(-1)[wi] for k, v in sR.items()}, eos, idir=idir,
            fallback=fallback, return_mask=True)
        for k in F:
            F[k].reshape(-1)[wi] = Fw[k]
            U[k].reshape(-1)[wi] = Uw[k]
        p_star.reshape(-1)[wi] = pw
        n_weak_linear = int(okw.sum())
        weak_linear[wi[okw.reshape(-1)]] = True
    # the fan lies entirely on one side: HLLD already returns the exact flux
    upwind = torch.zeros_like(weak)
    if _UPWIND_SKIP:
        *_, cmax_hl, cmin_hl = compute_srmhd_fluxes(sL, sR, eos, idir)
        upwind = (~bad) & (~weak) & ((cmin_hl <= 0.0) | (cmax_hl <= 0.0))
    live = to_np((~bad) & (~weak) & (~upwind)).astype(bool)
    # the compound rule, on the solver-frame tangential fields (see
    # _COMPOUND_SKIP): weak against the normal field, and reversing
    routed = np.zeros(N, dtype=bool)
    compound = np.zeros(N, dtype=bool)
    if _COMPOUND_SKIP or _PLANAR4_SKIP_COMPOUND:
        BtL = np.hypot(left[5], left[6])
        BtR = np.hypot(right[5], right[6])
        # the wrapped difference of the two field angles, NOT atan2 of the
        # cross and dot product: they agree except where one side's
        # tangential field is exactly zero, and those lanes (the rotor's
        # t = 0 x-sweeps) are 52 of the 252 the fitted rule flags
        psiL = np.arctan2(left[6], left[5])
        psiR = np.arctan2(right[6], right[5])
        dpsi = np.abs((psiR - psiL + _math.pi) % (2.0 * _math.pi) - _math.pi)
        compound = live & (np.minimum(BtL, BtR)
                           < _COMPOUND_BT_MAX * np.abs(Bn)) & (dpsi > _COMPOUND_DPSI_MIN)
    if _COMPOUND_SKIP:
        routed = compound
        live = live & ~routed
    sel = np.flatnonzero(live)

    diag = dict(solver="exact-batched", idir=idir, n_interfaces=N,
                n_bad=int(bad.sum()), n_weak_gate=int(weak.sum()),
                n_upwind_skip=int(upwind.sum()),
                n_compound_skip=int(routed.sum()))

    if sel.size == 0:
        if _rendezvous is not None:
            _rendezvous.skip(rec_idir)          # wait for this sweep's turn
        # exact_mask has to be here too: every consumer reads it
        # unconditionally (scripts/rotor_replay.py:104), and with the
        # compound routing on, a sweep CAN gate out every live lane -- the
        # rotor's t = 0 x-sweep does, where one side has |Bt| = 0 exactly
        LAST_DIAG.update(**diag, n_attempted=0, n_exact=0,
                         n_hlld_fallback=N, frac_hlld_fallback=1.0,
                         frac_exact=0.0, compound_mask=routed,
                         exact_mask=torch.zeros(N, dtype=torch.bool),
                         sel=sel, reason7=np.zeros(0, np.int8),
                         reason5=np.zeros(0, np.int8),
                         seven_attempts=np.zeros(0, int),
                         seven_n_iter_total=np.zeros(0, int),
                         planar_n_iter=np.zeros(0, int),
                         planar_rung=np.zeros(0, np.int8),
                         reason4=np.zeros(0, np.int8),
                         planar4_n_iter=np.zeros(0, int),
                         planar4_kinds=np.zeros((2, 0), np.int8),
                         n_planar4_attempted=0, n_planar4_exact=0,
                         planar4_mask=np.zeros(0, bool),
                         n_continuation_attempted=0, n_continuation_exact=0,
                         continuation_mask=np.zeros(0, bool),
                         n_weak=int(weak.sum()), n_weak_linear=n_weak_linear,
                         weak_linear_mask=weak_linear,
                         n_linear_fallback_offered=0, n_linear_fallback=0,
                         linear_fallback_mask=np.zeros(0, bool),
                         n_hlld_after_attempt=0)
        if harvester is not None:
            # nothing was attempted, so the map is all gate reasons -- which
            # is exactly what makes a sweep like the rotor's t = 0 x-sweep
            # legible afterwards
            harvester.record_coverage(rec_idir, N, np.zeros(0, dtype=int),
                                      np.zeros(0, dtype=int),
                                      np.flatnonzero(routed),
                                      bad=to_np(bad).astype(bool),
                                      weak=to_np(weak).astype(bool),
                                      upwind=to_np(upwind).astype(bool),
                                      jump=to_np(jump).reshape(-1))
        return F, U, p_star

    subL = [c[sel] for c in left]
    subR = [c[sel] for c in right]
    subBn = Bn[sel]

    # The ML warm start is not optional here.  Without it the Newton starts
    # from a structure-free guess, and the difference is not marginal: on the
    # same 40 interfaces, neutral seeding solved 23 and took 1896 s where the
    # warm start solved 30 in 383 s.  The scalar path auto-loads for the same
    # reason, so the batched one must too or the comparison is meaningless.
    from rmhd import ml_guess as mg
    if model is None or scaler is None:
        model, scaler = _load_model_cached(_rmhd_paths.resolve(_DEFAULT_CKPT))
    from rmhd.batched import ml_b as MB
    cands, feats = MB.predict_unk6_k(
        model, scaler, np.stack(subL, axis=1), np.stack(subR, axis=1), subBn)
    seed6 = cands[0]          # a best-of-K checkpoint: candidate 0 is the primary
    # The net is an interpolator; outside its training box it extrapolates
    # without limit.  Measured on the t=0 rotor x-sweep, where |Bt| is exactly
    # zero on both sides: it is fed log(0 + 1e-30) = -69 against a training
    # minimum of -6.9 and answers ln|Bt_CD| ~ 110, seeding the Newton at 1e48
    # so its first residual is inf.  Bounding the seed does not rescue those
    # lanes -- they fall back to HLLD and are counted -- it keeps a nonsense
    # seed from putting non-finite numbers into a shared batch.
    seed6, n_seed_clamped = MB.clamp_seed_physical(seed6, subL, subR, subBn)
    # per-lane RNG keys folded from each lane's own canonical features, so a
    # retry jitter cannot depend on where the interface sits in the batch
    keys = MB.lane_keys(feats) if n_retries else None
    # the ensemble: one clamped seed per extra checkpoint, retry k from the
    # k-th (per-lane predictions, so batch-independence is untouched)
    seeds_extra = None
    if n_retries and (len(cands) > 1 or _EXTRA_CKPTS):
        seeds_extra = []
        for u_k in cands[1:]:               # the primary's own extra candidates first
            u_k, _ = MB.clamp_seed_physical(u_k, subL, subR, subBn)
            seeds_extra.append(u_k)
        for ck in _EXTRA_CKPTS:
            m_k, s_k = _load_model_cached(_rmhd_paths.resolve(ck))
            u_k, _ = MB.predict_unk6(m_k, s_k, np.stack(subL, axis=1),
                                     np.stack(subR, axis=1), subBn)
            u_k, _ = MB.clamp_seed_physical(u_k, subL, subR, subBn)
            seeds_extra.append(u_k)

    # tau_bt is applied INSIDE solve_batch rather than here: it must only
    # skip the seven-wave classes.  The reduced three-wave solver carries no
    # rotation unknowns and handles Bt = 0 correctly -- 8 of the rotor's 12
    # degenerate x-sweep lanes converge on it at t=0 -- so a tier-0 gate would
    # have thrown those answers away along with the unrepresentable ones.
    run = (_run_pipeline if _rendezvous is None
           else lambda *a, **k: _rendezvous.pipeline(rec_idir, *a, **k))
    res, d, star, region, ray_ok, n_fan = run(
        P, gamma, subL, subR, subBn, seed6, keys, seeds_extra,
        accuracy=accuracy, max_iter=max_iter, n_retries=n_retries,
        tau_bt=tau_bt)
    diag.update(d)
    diag["n_seed_clamped"] = n_seed_clamped
    diag["n_ensemble_seeds"] = 0 if seeds_extra is None else len(seeds_extra)

    from rmhd.batched import classify as C
    cls = res["cls"]
    conv = res["converged"]

    # ── the resolved state must itself be physical ────────────────────────
    rho, P_tot, vn, vt1, vt2, Bt1, Bt2 = star
    v2 = vn * vn + vt1 * vt1 + vt2 * vt2
    W2 = 1.0 / np.maximum(1.0 - v2, 1e-300)
    eta = subBn * vn + Bt1 * vt1 + Bt2 * vt2
    b2 = (subBn ** 2 + Bt1 ** 2 + Bt2 ** 2) / W2 + eta ** 2
    physical = (rho > 0.0) & (v2 < 1.0) & (P_tot - 0.5 * b2 > 0.0)
    for c in star:
        physical &= np.isfinite(c)
    take = conv & ray_ok & physical

    n_unphys = int((conv & ray_ok & ~physical).sum())

    # ── admissibility: the waves must not cross ───────────────────────────
    # Applied to the seven-wave path only, and BEFORE the harvest.  It used to
    # run after the rescues, i.e. after the harvest too, so every fan it
    # rejected was still written to the solved shards as a training row.
    # MEASURED by replaying this function over the shards (2026-09-19): 0.53%
    # of the seven-wave rows of rotor_64_exact and 1.04% of rotor_128_exact
    # (24,238 of 2.33M) were self-crossing, 3-7% in the first steps.
    #
    # "Seven-wave path" means the seven-wave CLASSES, not `take`: the
    # degenerate classes are answered inside solve_batch by the three-wave
    # solver, which fills only R4/R5 and VsL/VsR.  Their VsLv/VsRv are zero
    # (or, after a reclassification, the abandoned seven-wave attempt's), so
    # the order read off them is [0, 0, 0, v_cd, 0, 0, 0] and every such lane
    # with |v_cd| > tol was rejected.  MEASURED over a 32^2 rotor to t=0.4:
    # 2205 of 2472 rejections (89%) were degenerate lanes, 1.7% of all exact
    # fluxes (72 against 69 in the first step).  The planar answers need no
    # test: both rotations have zero strength by construction, so they cannot
    # cross anything that matters.
    crossed = np.zeros(sel.size, dtype=bool)
    if _WAVE_ORDER and take.any():
        crossed = _self_crossing(res["zones"], res["VsLv"], res["VsRv"],
                                 _WAVE_ORDER_TOL)
        crossed &= take & np.isin(cls, (C.FULL7, C.COPLANAR))
        take = take & ~crossed
    inadm7 = np.zeros(sel.size, dtype=bool)
    if _ADMISSIBLE_GATE and take.any():
        inadm7 = take & np.isin(cls, (C.FULL7, C.COPLANAR)) & ~_admissible_structure(
            P, gamma, subL, subR, subBn, res["zones"], res["VsLv"], res["VsRv"])
        take = take & ~inadm7
        # offered to the rescues below exactly as a self-crossing lane
        crossed = crossed | inadm7
    n_crossed = int(crossed.sum())

    # Harvest before the flux assembly, while the per-lane verdict is still
    # in hand: `take` is the only place that knows a lane cleared EVERY gate
    # (converged, ray resolved, state physical, fan ordered), and downstream
    # only the assembled flux survives.
    if harvester is not None:
        from src.physics.harvest import Harvester           # noqa: F401
        harvester.record(
            subL, subR, subBn, cls=cls, converged=conv, zones=res["zones"],
            VsLv=res["VsLv"], VsRv=res["VsRv"],
            attempts=res.get("attempts", np.ones(sel.size, dtype=int)),
            accepted=take, ray_ok=ray_ok, crossed=crossed, source=0,
            seven_wave=np.isin(cls, (C.FULL7, C.COPLANAR)))

    # ── degenerate answers: only where the INPUT was degenerate ───────────
    # solve_batch re-solves with the three-wave solver any lane whose
    # seven-wave attempt came out degenerate at R4/R5.  That class is read off
    # a possibly unconverged attempt, and the three-wave answer then neglects
    # waves the input needs.  MEASURED 2026-09-19 against the verified planar
    # solve on the 64^2 rotor: those lanes are where the three-wave flux is
    # worst (0.55-0.82 relative at |B_n| ~ 1, where HLLD is off 0.19), while
    # on the 349 lanes degenerate from the input it is off 4e-5 (median),
    # 1.8e-2 (max) against HLLD's 5.9e-3 and 8.8.  Refused here -- after the
    # harvest, which would record a refused physical lane as "unphysical",
    # and before the planar rescue, which may still answer it exactly.
    reclass = np.asarray(res["reclassified"], dtype=bool)
    n_reclass_refused = int((take & reclass).sum())
    # Why the seven-wave path lost a lane, one code per attempted lane --
    # diagnostics only, nothing below reads it: 0 accepted, 1 the Newton did
    # not converge, 2 the ray could not be resolved, 3 unphysical state,
    # 4 self-crossing fan, 5 degenerate answer refused (reclassified),
    # 6 refused by the admissibility gate.
    reason7 = np.zeros(sel.size, dtype=np.int8)
    reason7[take & reclass] = 5
    reason7[crossed] = 4
    reason7[inadm7] = 6
    reason7[conv & ray_ok & ~physical] = 3
    reason7[conv & ~ray_ok] = 2
    reason7[~conv] = 1
    # The harvest above counted these lanes answered, so it wrote them nowhere
    # (a degenerate class is not a solved row, an accepted lane not an
    # unsolved one), yet they go on to the rescues below.  MEASURED
    # 2026-11-06: ~18.9k of the 162k lanes the 4Q 64^2 run's Newton left to
    # the rescues (attempted - seven - degenerate = 162,354 against 143,502
    # unsolved rows).  With RMHD_HARVEST_RESCUES they are unsolved rows too.
    if harvester is not None and _HARVEST_RESCUES and n_reclass_refused:
        from src.physics.harvest import R_RECLASSIFIED
        harvester.record_unsolved(
            subL, subR, subBn, np.flatnonzero(take & reclass), reason=R_RECLASSIFIED,
            cls=cls, attempts=res.get("attempts", np.ones(sel.size, dtype=int)))
    take = take & ~reclass

    # A self-crossing lane is NOT offered to the rescues below (hence the
    # `~crossed` in both candidate masks).  That is what the code did while
    # the gate ran after them, so moving the gate changes the harvest and
    # nothing else.  Offering them to the planar solver would change the
    # flux, which is a separate decision.

    # ── reduced three-wave rescue (opt-in) ────────────────────────────────
    # Only lanes the seven-wave solver did NOT win, and only where the
    # normal field is weak enough for the neglected slow waves to cost less
    # than a part in a thousand.  Runs after the harvest so the harvester's
    # `accepted` stays strictly seven-wave -- the reduced solution has no
    # Alfven or slow zones and would not fit the solved-shard schema -- but
    # before the flux assembly, so the run gets the better flux.
    take3 = np.zeros(sel.size, dtype=bool)
    n3_att = 0
    if _3WAVE_FALLBACK:
        cand = np.flatnonzero(~take & ~crossed & (np.abs(subBn) <= _BN_3WX_MAX))
        n3_att = int(cand.size)
        if cand.size:
            r3 = P["reduced"]([c[cand] for c in subL], [c[cand] for c in subR],
                              subBn[cand], 0, accuracy=accuracy)
            c3 = r3["converged"].astype(bool)
            st3, reg3, e3 = RAY.state_at_xi_3wave(
                [c[cand] for c in subL], [c[cand] for c in subR],
                r3["solutionL"], r3["solutionR"], r3["VsL"], r3["VsR"],
                subBn[cand], gamma, P["xi"], P["fan_p"])
            good = c3 & ~e3
            if good.any():
                rho3, P3, vn3, vt13, vt23, Bt13, Bt23 = st3
                v2_ = vn3 * vn3 + vt13 * vt13 + vt23 * vt23
                W2_ = 1.0 / np.maximum(1.0 - v2_, 1e-300)
                eta_ = subBn[cand] * vn3 + Bt13 * vt13 + Bt23 * vt23
                b2_ = (subBn[cand] ** 2 + Bt13 ** 2 + Bt23 ** 2) / W2_ + eta_ ** 2
                phys3 = (rho3 > 0.0) & (v2_ < 1.0) & (P3 - 0.5 * b2_ > 0.0)
                for c_ in st3:
                    phys3 &= np.isfinite(c_)
                good &= phys3
            if good.any():
                w = cand[good]
                for j in range(7):
                    star[j][w] = st3[j][good]
                region[w] = reg3[good]
                take3[w] = True

    # ── five-wave PLANAR rescue (opt-in) ──────────────────────────────────
    # Same placement as the three-wave block: after the harvest, so the
    # harvested rows stay exactly what they were, and before the flux
    # assembly, so the run gets the better flux.  The answers are mapped into
    # the seven-wave zone layout -- R2 == R3 and R6 == R7, the two rotational
    # discontinuities having zero strength -- so the EXISTING ray sampler
    # resolves xi = 0 with no new code.  That mapping is exact, not a
    # convenience: a five-wave solution IS a seven-wave solution whose
    # rotations vanish, which is what `planar5_b.solve` has already verified.
    # the planarity of every lane, once, for the planar rungs' pre-check
    planar_ok = np.ones(sel.size, dtype=bool)
    n_nonplanar_skipped = 0
    if _PLANAR_PRECHECK and (_PLANAR5_FALLBACK or _PLANAR4) and sel.size:
        from rmhd.batched import planar5_b as _P5B
        planar_ok = np.asarray(_P5B.to_planar(subL, subR, subBn)[3]) <= _PLANAR_TOL
    take5 = np.zeros(sel.size, dtype=bool)
    n5_att = 0
    reason5 = np.full(sel.size, 7, dtype=np.int8)        # 7 = not offered
    iters5 = np.full(sel.size, -1, dtype=int)
    rung5 = np.full(sel.size, -1, dtype=np.int8)         # -1 = not by planar
    if _PLANAR5_FALLBACK:
        offered = ~take & ~take3
        if not _PLANAR5_CROSSED:
            offered &= ~crossed
        if _PLANAR_PRECHECK:
            skip = offered & ~planar_ok
            reason5[skip] = 2                               # not planar
            iters5[skip] = 0
            n_nonplanar_skipped = int(skip.sum())
            offered &= planar_ok
        cand = np.flatnonzero(offered)
        n5_att = int(cand.size)
        if cand.size:
            sL5 = [c[cand] for c in subL]
            sR5 = [c[cand] for c in subR]
            B5 = subBn[cand]
            # the rescues run over the worker pool when one is configured
            # (`exact_pool.map_lanes`, chunked by lane like the seven-wave
            # solve): measured on the 64^2 rotor with every rung on, the
            # rescues in the main process left 5.9 of 64 cores busy and cost
            # 8x the wall of the run without them (docs 5a)
            from . import exact_pool
            r5 = exact_pool.map_lanes("_planar_and_ray", gamma, sL5, sR5, B5,
                                      accuracy=accuracy, max_iter=max_iter)
            if _PLANAR5_LADDER and not r5["good"].all():
                r5 = exact_pool.map_lanes("_planar_ladder", gamma, sL5, sR5, B5,
                                          r5=r5, accuracy=accuracy, max_iter=max_iter)
                rung5[cand] = r5["rung"]
            else:
                rung5[cand] = np.where(r5["good"], 0, -1)
            good = r5["good"]
            reason5[cand] = r5["reason"]
            iters5[cand] = r5["n_iter"]
            if good.any():
                st5, reg5 = r5["star"], r5["region"]
                # A2 is A and D2 is D unless a rung put a reversal there
                A, A2, B_, C_, D2, D = r5["zones6"]
                VsLv5, VsRv5 = r5["VsLv"], r5["VsRv"]
                if good.any():
                    w = cand[good]
                    for j in range(7):
                        star[j][w] = st5[j][good]
                    region[w] = reg5[good]
                    take5[w] = True
                    # Harvest them as seven-wave rows -- which is what they
                    # are, with two zero-strength rotational discontinuities
                    # (R2 == R3, R6 == R7), already verified against the full
                    # six-residual inside `solve`.  Tagged source=1 so a
                    # trainer can weight or ablate rows whose rotations are
                    # identically zero.  Restricted to `w`: `record`'s
                    # unsolved branch fires on every call, so a second pass
                    # over the whole sweep would re-record the seven-wave
                    # successes as failures.
                    if harvester is not None and _HARVEST_PLANAR5:
                        k5 = int(good.sum())
                        hL = [c[good] for c in sL5]
                        hR = [c[good] for c in sR5]
                        hz = [[c[good] for c in z]
                              for z in (A, A2, B_, C_, D2, D)]
                        harvester.record(
                            hL, hR, B5[good],
                            cls=np.full(k5, C.COPLANAR, dtype=np.int8),
                            converged=np.ones(k5, dtype=bool),
                            zones=hz,
                            VsLv=[v[good] for v in VsLv5],
                            VsRv=[v[good] for v in VsRv5],
                            attempts=np.full(k5, 2, dtype=int),
                            accepted=np.ones(k5, dtype=bool),
                            seven_wave=np.ones(k5, dtype=bool),
                            source=1)

    # ── the planar seven-wave rung (RMHD_COPLANAR7=late) ──────────────────
    # After the five-wave solver, which answers most of the planar lanes the
    # Newton lost in a few iterations, and before the four-unknown one: the
    # COPLANAR lanes neither answered admissibly, from the seeds the sweep's
    # Newton started from.  Harvested as rows (source=2) only with
    # RMHD_HARVEST_RESCUES; the coverage map flags them either way.
    take7p = np.zeros(sel.size, dtype=bool)
    n7p_att = 0
    reason7p = np.full(sel.size, 8, dtype=np.int8)       # 8 = not offered
    stage7p = np.zeros(sel.size, dtype=np.int8)
    rot7p = np.zeros((2, sel.size), dtype=bool)
    iters7p = np.full(sel.size, -1, dtype=int)
    if _COPLANAR7_LATE:
        offered = ~take & ~take3 & ~take5 & planar_ok & (cls == C.COPLANAR)
        if not _PLANAR5_CROSSED:
            offered &= ~crossed
        cand = np.flatnonzero(offered)
        n7p_att = int(cand.size)
        if cand.size:
            from . import exact_pool
            seeds7 = dict(
                seed6=[np.asarray(c)[cand] for c in seed6],
                extra=(None if seeds_extra is None
                       else [[np.asarray(c)[cand] for c in s] for s in seeds_extra]))
            r7p = exact_pool.map_lanes("_coplanar7_and_ray", gamma,
                                       [c[cand] for c in subL],
                                       [c[cand] for c in subR], subBn[cand],
                                       seeds=seeds7, accuracy=accuracy)
            reason7p[cand] = r7p["reason"]
            stage7p[cand] = r7p["stage"]
            iters7p[cand] = r7p["n_iter"]
            rot7p[:, cand] = np.stack([np.asarray(x, dtype=bool) for x in r7p["rot"]])
            good = np.asarray(r7p["good"], dtype=bool)
            if good.any():
                w = cand[good]
                for j in range(7):
                    star[j][w] = r7p["star"][j][good]
                region[w] = np.asarray(r7p["region"])[good]
                take7p[w] = True
                if harvester is not None and _HARVEST_RESCUES:
                    _harvest_rescue(harvester, C, subL, subR, subBn, w, good, r7p,
                                    source=2,
                                    steer=np.stack([np.asarray(s)[good] for s in r7p["steer"]]),
                                    rot=np.stack([np.asarray(x, dtype=bool)[good] for x in r7p["rot"]]),
                                    stage=np.asarray(r7p["stage"])[good])

    # ── the four-unknown planar solver (production default) ──────────────
    # Last: only the lanes every rescue above lost.  Its certificate is its
    # own (each shock's jump conditions, the contact, the fan's order), not
    # the seven-wave residual, which cannot judge these answers -- see
    # planar4_b's module doc.
    take4 = np.zeros(sel.size, dtype=bool)
    n4_att = 0
    reason4 = np.full(sel.size, 8, dtype=np.int8)        # 8 = not offered
    iters4 = np.full(sel.size, -1, dtype=int)
    kinds4 = np.full((2, sel.size), -1, dtype=np.int8)
    n4_skipped = 0
    if _PLANAR4:
        offered = ~take & ~take3 & ~take5 & ~take7p
        if not _PLANAR5_CROSSED:
            offered &= ~crossed
        if _PLANAR4_SKIP_COMPOUND:
            n4_skipped = int((offered & compound[sel]).sum())
            offered &= ~compound[sel]
        if _PLANAR_PRECHECK:
            skip = offered & ~planar_ok
            reason4[skip] = 2                               # not planar
            iters4[skip] = 0
            n_nonplanar_skipped = max(n_nonplanar_skipped, int(skip.sum()))
            offered &= planar_ok
        cand = np.flatnonzero(offered)
        n4_att = int(cand.size)
        if cand.size:
            from . import exact_pool
            kw4 = dict(accuracy=accuracy, max_iter=max_iter)
            if _PLANAR4_STAR:
                kw4["hint"] = {"p": p_hll[np.asarray(sel)[cand]]}
            r4 = exact_pool.map_lanes("_planar4_and_ray", gamma,
                                      [c[cand] for c in subL],
                                      [c[cand] for c in subR], subBn[cand],
                                      **kw4)
            reason4[cand] = r4["reason"]
            iters4[cand] = r4["n_iter"]
            kinds4[:, cand] = r4["kinds"]
            good = r4["good"]
            if good.any():
                w = cand[good]
                for j in range(7):
                    star[j][w] = r4["star"][j][good]
                region[w] = np.asarray(r4["region"])[good]
                take4[w] = True
                if harvester is not None and _HARVEST_RESCUES:
                    # a rotation answer's R3/R6 differ from R2/R7 (_planar4_take)
                    z4 = r4["zones6"]
                    rot4 = np.stack([(np.asarray(z4[1][5]) != np.asarray(z4[0][5]))
                                     | (np.asarray(z4[1][6]) != np.asarray(z4[0][6])),
                                     (np.asarray(z4[4][5]) != np.asarray(z4[5][5]))
                                     | (np.asarray(z4[4][6]) != np.asarray(z4[5][6]))])
                    _harvest_rescue(harvester, C, subL, subR, subBn, w, good, r4,
                                    source=3,
                                    steer=np.asarray(r4["kinds"])[:, good].astype(np.int8),
                                    rot=rot4[:, good],
                                    stage=np.zeros(int(np.sum(good)), dtype=np.int8))

    # ── the continuation rung (opt-in) ────────────────────────────────────
    # Last: non-planar lanes every rung above lost, the seven-wave refusals
    # (self-crossing fans) included. Its first rung is seeded here, where the
    # networks live; the walk runs on the worker pool.
    takec = np.zeros(sel.size, dtype=bool)
    nc_att = 0
    reasonc = np.full(sel.size, 9, dtype=np.int8)        # 9 = not offered
    itersc = np.full(sel.size, -1, dtype=int)
    if _CONTINUATION:
        if not (_PLANAR_PRECHECK and (_PLANAR5_FALLBACK or _PLANAR4)) and sel.size:
            from rmhd.batched import planar5_b as _P5B
            planar_ok = np.asarray(_P5B.to_planar(subL, subR, subBn)[3]) <= _PLANAR_TOL
        offered = ~take & ~take3 & ~take5 & ~take7p & ~take4 & ~planar_ok
        cand = np.flatnonzero(offered)
        nc_att = int(cand.size)
        if cand.size:
            from . import continuation as CT
            from . import exact_pool
            sLc = [c[cand] for c in subL]
            sRc = [c[cand] for c in subR]
            Bc = subBn[cand]
            Lt, Rt = CT.rung0_problem(sLc, sRc, Bc)
            seeds = _ml_seeds(model, scaler, Lt, Rt, Bc, n_retries)
            rc = exact_pool.map_lanes("_continuation_and_ray", gamma, sLc, sRc,
                                      Bc, seeds=seeds, accuracy=accuracy,
                                      max_iter=max_iter, n_retries=n_retries)
            reasonc[cand] = rc["reason"]
            itersc[cand] = rc["n_iter"]
            good = np.asarray(rc["good"], bool)
            if good.any():
                w = cand[good]
                for j in range(7):
                    star[j][w] = rc["star"][j][good]
                region[w] = np.asarray(rc["region"])[good]
                takec[w] = True

    take_all = take | take3 | take5 | take7p | take4 | takec
    g = sel[take_all]
    if g.size:
        tt = lambda a: torch.tensor(a[take_all], dtype=dt)
        one = {k: v[g] for k, v in sL.items()}
        star_t = from_solver_frame([tt(c) for c in star], tt(subBn),
                                   eos, idir, one)
        uS, _, fS, _, _, _ = compute_srmhd_fluxes(star_t, star_t, eos, idir)
        for k in F:
            F[k][g] = fS[k]
            U[k][g] = uS[k]
        p_star[g] = torch.tensor(P_tot[take_all], dtype=dt)

    # ── a lost weak face: the linearised flux, not HLLD ──────────────────
    # (RMHD_WEAK_FALLBACK_TAU above.)  Only where the linearised solver
    # trusts its own answer (`ok`: a real spectrum, an invertible eigenvector
    # matrix, a physical state); the rest keep HLLD, which is already there.
    lin_fb = np.zeros(sel.size, dtype=bool)          # over the attempted lanes
    n_lin_offered = 0
    if _WEAK_FLUX == "linear" and _WEAK_FALLBACK_TAU > 0.0:
        jump_sel = to_np(jump).reshape(-1)[sel]
        cand = np.flatnonzero(~take_all & (jump_sel < _WEAK_FALLBACK_TAU))
        n_lin_offered = int(cand.size)
        if cand.size:
            from .linearised import linearised_flux
            ci = torch.as_tensor(sel[cand], dtype=torch.long)
            Fl, Ul, pl, okl = linearised_flux(
                {k: v.reshape(-1)[ci] for k, v in sL.items()},
                {k: v.reshape(-1)[ci] for k, v in sR.items()}, eos, idir=idir,
                fallback=fallback, return_mask=True)
            okl = torch.as_tensor(okl).reshape(-1).to(torch.bool)
            w = ci[okl]
            for k in F:
                F[k].reshape(-1)[w] = Fl[k].reshape(-1)[okl]
                U[k].reshape(-1)[w] = Ul[k].reshape(-1)[okl]
            p_star.reshape(-1)[w] = pl.reshape(-1)[okl]
            lin_fb[cand[okl.numpy()]] = True

    # ── verification: what makes a flux "exact" ───────────────────────────
    # A flux is called exact iff the structure behind it satisfies the
    # COMPLETE seven-wave jump conditions, checked per interface, in the
    # frame the answer was produced in (the residual is frame-sensitive, and
    # the solver frame's orientation is set arbitrarily by the sweep
    # direction).  Wave count is not the criterion: a five-wave planar answer
    # passes with a median residual of 1.1e-10, while the reduced three-wave
    # answer never reaches 1e-6 on the same test and is therefore not exact
    # however reliably it converges.
    #
    # This makes the paper's central claim a measured column rather than an
    # argument: `n_verified` is reported beside `n_exact`, and they must
    # agree.  The planar path already verifies internally, so what is added
    # here is the seven-wave path and the run-level number.
    verified = np.zeros(sel.size, dtype=bool)
    if take.any() and _VERIFY_EXACT:
        f6 = P["planar5"]["waves"]["fullfuncv6"]
        seven = np.isin(cls, (C.FULL7, C.COPLANAR))

        def _check(w, unk6):
            if w.size == 0:
                return
            fvv, _, _, _, ev = f6([c[w] for c in subL], [c[w] for c in subR],
                                  [u for u in unk6], subBn[w])
            nv = np.where(ev, np.inf, np.max(np.abs(fvv), axis=0))
            verified[w] = nv <= _VERIFY_TOL

        # the seven-wave path fills `unk` directly
        _check(np.flatnonzero(take & seven),
               [u[np.flatnonzero(take & seven)] for u in res["unk"]])

        # The DEGENERATE classes are answered by the reduced three-wave
        # solver inside `solve_batch`, which fills only R4 and R5 and never
        # touches `unk` -- so they must be rebuilt from the structure or the
        # check is meaningless (it would evaluate at all-zero unknowns and
        # reject every one of them).  A three-wave answer is a seven-wave one
        # with R2 = R3 = R4 and R5 = R6 = R7, so both post-fast pressures are
        # the contact pressure and neither rotation turns.
        d = np.flatnonzero(take & ~seven)
        if d.size:
            R4 = [c[d] for c in res["zones"][2]]
            Bt4 = np.hypot(R4[5], R4[6])
            z = np.zeros(d.size)
            _check(d, [np.log(np.maximum(R4[1], 1e-30)),
                       np.log(np.maximum(Bt4, 1e-30)),
                       np.arctan2(R4[6], R4[5]),
                       np.log(np.maximum(R4[1], 1e-30)), z, z])
    # the three-wave path is deliberately never marked verified (measured:
    # 0.0% of its answers reach 1e-6 on this residual); the planar path
    # verifies inside `solve` before it reports convergence
    verified |= take5
    # the four-unknown path certifies inside `solve` (each shock's
    # Rankine-Hugoniot conditions to the same tolerance, the contact, the
    # fan's order); the seven-wave residual is not its certificate
    verified |= take4
    # the planar seven-wave rung certifies as the four-unknown one does (it
    # IS that solver, with rotations), and the gate has read every shock
    verified |= take7p
    # the continuation verifies inside the rung: the full seven-wave residual
    # to the same tolerance, in the planar frame it was produced in
    verified |= takec
    # Degenerate-class lanes answered by solve_batch's three-wave solver are
    # counted apart: exact in the limit their class names (B_n -> 0, or an
    # Alfven wave merged with a magnetosonic one), not verified -- 3 of ~2000
    # reach 1e-8 on the seven-wave residual.  So on a run with degenerate
    # interfaces n_exact = n_verified + (n_degenerate_exact -
    # n_degenerate_verified) [+ the opt-in n_3wave_exact], by design.
    degen = take & ~np.isin(cls, (C.FULL7, C.COPLANAR))

    # The complete per-face map, recorded here rather than before the flux
    # assembly so it can carry `verified` too.  Every mask below is over the
    # ATTEMPTED subset except the three gate reasons, which are over the whole
    # sweep; `record_coverage` scatters the first kind back.  This is what
    # `scripts/plot_rotor_output.py` turns into "which solver answered this
    # cell".
    if harvester is not None:
        seven_ok = take & np.isin(cls, (C.FULL7, C.COPLANAR))
        harvester.record_coverage(
            rec_idir, N, sel, g, np.flatnonzero(routed),
            seven=seven_ok, planar5=take5, three_wave=take3, planar4=take4,
            continuation=takec, coplanar7=take7p, degenerate=degen, verified=verified,
            linear_fallback=lin_fb,
            bad=to_np(bad).astype(bool), weak=to_np(weak).astype(bool),
            upwind=to_np(upwind).astype(bool), jump=to_np(jump).reshape(-1))

    n_exact = int(g.size)
    fell_back = N - n_exact
    exact_mask = torch.zeros(N, dtype=torch.bool)
    if g.size:
        exact_mask[torch.as_tensor(g, dtype=torch.long)] = True
    LAST_DIAG.update(
        **diag,
        exact_mask=exact_mask,
        compound_mask=routed,
        # per attempted lane (index into the sweep: `sel`): why each solver
        # lost it, and what it cost
        sel=sel, reason7=reason7, reason5=reason5,
        seven_attempts=np.asarray(res.get("attempts", np.ones(sel.size, int))),
        seven_n_iter_total=np.asarray(
            res.get("n_iter_total", res.get("n_iter", np.zeros(sel.size, int)))),
        # RMHD_COPLANAR7: the seven-wave answers that came from the planar
        # form, its stage (1 seed's rotations, 2 other rotations, 3 extra
        # seeds) and the rotations it put in (2, n_attempted)
        coplanar7_mask=np.asarray(res.get("coplanar7", np.zeros(sel.size, bool))),
        coplanar7_stage=np.asarray(res.get("c7_stage", np.zeros(sel.size, np.int8))),
        coplanar7_rot=np.stack([np.asarray(x) for x in res.get(
            "c7_rot", [np.zeros(sel.size, bool), np.zeros(sel.size, bool)])]),
        # RMHD_COPLANAR7=late: the same solver as a rescue rung after the
        # five-wave one -- which lanes it answered, why it refused the others
        # (8 not offered), its stage, rotations and iterations
        n_coplanar7_late_attempted=n7p_att,
        n_coplanar7_late_exact=int(take7p.sum()),
        coplanar7_late_mask=take7p, coplanar7_late_reason=reason7p,
        coplanar7_late_stage=stage7p, coplanar7_late_rot=rot7p,
        coplanar7_late_n_iter=iters7p,
        planar_n_iter=iters5,
        # which start of the planar rescue answered: 0 its default, k the
        # k-th rung of RMHD_PLANAR5_LADDER, -1 not answered by the rescue
        planar_rung=rung5,
        # the four-unknown rung: why each offered lane was refused, its
        # iterations, and how each slow wave was steered (0 field, 1 pressure)
        reason4=reason4, planar4_n_iter=iters4, planar4_kinds=kinds4,
        n_planar4_attempted=n4_att,
        n_planar4_exact=int(take4.sum()),
        planar4_mask=take4,
        n_planar4_skipped_compound=n4_skipped,
        # lanes the planar rungs were not offered because their input is not
        # planar (RMHD_PLANAR_PRECHECK; none of them could have been accepted)
        n_planar_skipped_nonplanar=n_nonplanar_skipped,
        # the continuation rung (RMHD_CONTINUATION): offered, answered, why
        # each offered lane was refused (continuation.REASONS; 9 = not
        # offered), its Newton iterations over the whole walk
        n_continuation_attempted=nc_att,
        n_continuation_exact=int(takec.sum()),
        continuation_mask=takec, reason_cont=reasonc,
        continuation_n_iter=itersc,
        # faces below the weak-jump gate, and how many of them the linearised
        # solver answered (RMHD_WEAK_FLUX=linear; 0 with HLLD there)
        n_weak=int(weak.sum()), n_weak_linear=n_weak_linear,
        weak_linear_mask=weak_linear,
        # attempted faces every exact rung lost: how many were weak enough
        # for the linearised fallback (RMHD_WEAK_FALLBACK_TAU), how many it
        # answered (mask over the attempted lanes), and how many were left
        # to HLLD.  n_hlld_fallback below counts every non-exact face of the
        # sweep (gated ones included) and is kept for its old meaning.
        n_linear_fallback_offered=n_lin_offered,
        n_linear_fallback=int(lin_fb.sum()),
        linear_fallback_mask=lin_fb,
        n_hlld_after_attempt=int(sel.size) - n_exact - int(lin_fb.sum()),
        n_attempted=int(sel.size),
        n_exact=n_exact,
        n_fan_interior=n_fan,
        n_unphysical_star=n_unphys,
        n_ray_failed=int((conv & ~ray_ok).sum()),
        n_3wave_attempted=n3_att,
        n_3wave_exact=int(take3.sum()),
        three_wave_mask=take3,
        n_planar5_attempted=n5_att,
        n_planar5_exact=int(take5.sum()),
        planar5_mask=take5,
        n_verified=int(verified[take_all].sum()),
        n_wave_order_rejected=n_crossed,
        verified_mask=verified,
        n_degenerate_exact=int(degen.sum()),
        n_degenerate_verified=int((degen & verified).sum()),
        degenerate_mask=degen,
        n_reclassified_refused=n_reclass_refused,
        n_hlld_fallback=fell_back,
        frac_hlld_fallback=fell_back / max(N, 1),
        frac_exact=n_exact / max(N, 1),
    )
    return F, U, p_star
