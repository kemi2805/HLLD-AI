"""
The batched exact Riemann flux: agreement, invariants, and cost.

Separate from ``test_idir.py``'s parametrized sweep rather than added to it,
for two reasons worth stating: that test draws 4000 wild random states, which
for the exact solver means a large fallback fraction and minutes per call, and
its ``frac < 0.005`` branch-flip bound is calibrated for HLLD's HLLE fallback,
not for a solver whose fallback predicate is a Newton convergence flag.  The
same PROPERTY is asserted here at a size the exact solver can afford, with the
bound set from what this flux actually does.

Ground truth is the forward-constructed solution, so scalar-vs-batched
agreement checks the port while agreement with truth checks the physics --
two different questions that a single comparison would conflate.

Sizing.  These tests run at N = 20 with ``max_iter = 10``, which is a
deliberate choice rather than a default.  The crossover measurement puts a
small batch in the worst possible regime -- 37.9 s per interface at N = 40
against 1.16 s at N = 4000 -- so the suite as first written would have taken
about 3.5 hours.  Every property asserted here is STRUCTURAL (rotation
equivalence, batch independence, agreement between two implementations), and
none of them depends on how many Newton iterations are allowed, provided both
sides of a comparison get the same budget.  A smaller budget converges fewer
lanes; it does not change the answer on the lanes that do converge, and each
test checks it has enough of those left to conclude anything.
"""

N_PROB = 20
MAX_ITER = 10
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_RMHD = "/Users/miler/Codes/rmhd_final"
if _RMHD not in sys.path:
    sys.path.append(_RMHD)

from src.physics.eos import hybrid_eos                       # noqa: E402
from src.physics.hlld import LAST_DIAG, hlld_flux            # noqa: E402
from src.physics.exact_flux import (exact_flux,              # noqa: E402
                                    exact_flux_batched)

GAMMA = 5.0 / 3.0
VEC = [("vx", "vy", "vz"), ("Bx", "By", "Bz")]
CONS = [("Sx", "Sy", "Sz"), ("Bx", "By", "Bz")]


@pytest.fixture(scope="module")
def eos():
    return hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)


@pytest.fixture(scope="module")
def problems(eos):
    """Forward-constructed solutions whose fan straddles the interface ray."""
    from batched.ray_scalar import claim_rmhd_namespace
    claim_rmhd_namespace()
    from eos import set_eos
    set_eos("ideal")
    from riemann_dataset import generate_dataset

    sols = generate_dataset(N_PROB, gamma=GAMMA, Bx_range=(0.3, 3.0), seed=5,
                            xi_window=(0.0, 0.0), verbose=False)
    Z = np.array([s.zones for s in sols])
    Bx = np.array([s.Bx for s in sols])

    def prim(slot):
        rho, Pt, vn, v1, v2_, B1, B2 = (Z[:, slot, j] for j in range(7))
        v2 = vn ** 2 + v1 ** 2 + v2_ ** 2
        W2 = 1.0 / (1.0 - v2)
        eta = Bx * vn + B1 * v1 + B2 * v2_
        b2 = (Bx ** 2 + B1 ** 2 + B2 ** 2) / W2 + eta ** 2
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        d = {"rho": t(rho), "vx": t(vn), "vy": t(v1), "vz": t(v2_),
             "p": t(Pt - 0.5 * b2), "Bx": t(Bx), "By": t(B1), "Bz": t(B2)}
        d["eps"] = eos.eps__press_rho(d["p"], d["rho"])
        return d

    return prim(0), prim(7), len(sols)


def cyc(d, triples, n):
    out = dict(d)
    for tr in triples:
        vals = [d[k] for k in tr]
        for i, k in enumerate(tr):
            out[k] = vals[(i - n) % 3]
    return out


def test_uniform_state_gives_the_physical_flux(eos, problems):
    """L == R has no waves: the flux must be F(U_L), exactly.

    ``tau_weak = 0.0`` is essential here, not incidental.  With L == R the
    relative jump is exactly zero, so with the normal gate every lane would
    take the HLLD fallback and this would silently test HLLD instead of the
    exact solver.  Disabling the gate forces the exact path to run, and then
    every ray lands in a constant region -- which makes this the cheapest
    test that can catch a region-indexing bug, since the answer is known
    without solving anything.
    """
    sL, _, _ = problems
    F, U, p = exact_flux_batched(sL, sL, eos, idir=0, tau_weak=0.0,
                                   max_iter=MAX_ITER)
    from src.physics.hlld import compute_srmhd_fluxes
    uref, _, fref, _, _, _ = compute_srmhd_fluxes(sL, sL, eos, idir=0)
    for k in F:
        sc = torch.maximum(fref[k].abs().max(),
                           torch.tensor(1e-30, dtype=torch.float64))
        # 1e-8, not machine precision: with the weak gate disabled the solver
        # actually WORKS on the trivial problem and converges to `accuracy`
        # (1e-8) in the RESIDUAL, which lands ~2e-10 in the state.  Demanding
        # exactness here asks for more than the solver is configured to give.
        # A region-indexing bug -- what this test is for -- shows up at O(1).
        assert float((F[k] - fref[k]).abs().max() / sc) < 1e-8, (
            f"uniform state gave a non-physical flux in {k}")


def test_matches_the_scalar_reference(eos, problems):
    """The port check: same flux as the looping scalar implementation."""
    sL, sR, n = problems
    Fb, _, pb = exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    mb = LAST_DIAG["exact_mask"].clone()
    Fs, _, ps = exact_flux(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    ms = LAST_DIAG["exact_mask"].clone()
    # p* > 0 does NOT mean "solved exactly" -- hlld_flux writes a positive p*
    # on every lane IT solved, so the sign only marks HLLD's own fallback.
    both = mb & ms
    assert int(both.sum()) >= n // 3, (
        f"only {int(both.sum())}/{n} interfaces solved exactly by both")
    worst = 0.0
    for k in Fb:
        a, b = Fs[k][both], Fb[k][both]
        sc = torch.maximum(a.abs(), b.abs()).clamp(min=1e-30)
        worst = max(worst, float(((a - b).abs() / sc).max()))
    assert worst < 1e-8, f"batched flux differs from scalar by {worst:.3e}"


@pytest.mark.parametrize("n_cyc", [1, 2], ids=["idir1", "idir2"])
def test_rotation_equivalence(eos, problems, n_cyc):
    """flux(C^n P, idir=n) must equal C^n(flux(P, idir=0)).

    Interfaces where the two runs take different discrete branches are
    excluded from the value comparison and bounded by count, as in
    test_idir.py: the exact solver's fallback predicate is a Newton
    convergence flag, and a problem sitting on the edge of the basin can
    converge along x and not along the rotated axis.  That is not a direction
    bug -- the same fragility exists along x -- so it is required to be rare,
    not absent.
    """
    sL, sR, n = problems
    f_ref, _, ps_ref = exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    m_ref = LAST_DIAG["exact_mask"].clone()
    f_rot, _, ps_rot = exact_flux_batched(cyc(sL, VEC, n_cyc),
                                          cyc(sR, VEC, n_cyc), eos,
                                          idir=n_cyc, max_iter=MAX_ITER)
    m_rot = LAST_DIAG["exact_mask"].clone()
    f_expect = cyc(f_ref, CONS, n_cyc)

    flip = m_ref ^ m_rot
    keep = ~flip
    assert int(keep.sum()) >= n // 3, "too few same-branch interfaces to judge"
    worst = 0.0
    for k in f_ref:
        a, b = f_expect[k][keep], f_rot[k][keep]
        sc = torch.maximum(a.abs().max(), torch.tensor(1e-30,
                                                       dtype=a.dtype))
        worst = max(worst, float((a - b).abs().max() / sc))
    assert worst < 1e-10, (
        f"exact_flux_batched is not idir-generic (idir={n_cyc}): {worst:.3e}")
    frac = float(flip.sum()) / n
    assert frac < 0.10, (
        f"convergence decision flips on {frac*100:.1f}% of interfaces under "
        f"rotation -- too many to be basin-edge marginality")


def test_batch_independence(eos, problems):
    """A face's flux must not depend on which faces share its batch.

    Required by domain decomposition: under MPI the same interface will sit
    in a different batch on a different rank, and the answer must not move.
    """
    sL, sR, n = problems
    F, _, p = exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    half = n // 2
    for sl in (slice(0, half), slice(half, None)):
        a = {k: v[sl] for k, v in sL.items()}
        b = {k: v[sl] for k, v in sR.items()}
        Fp, _, pp = exact_flux_batched(a, b, eos, idir=0,
                                       max_iter=MAX_ITER)
        assert torch.equal(pp, p[sl]), "splitting the batch changed p*"
        for k in F:
            assert torch.equal(Fp[k], F[k][sl]), (
                f"splitting the batch changed the flux in {k}")


def test_diagnostics_account_for_every_interface(eos, problems):
    """Every face is either solved exactly or reported as a fallback."""
    sL, sR, n = problems
    exact_flux_batched(sL, sR, eos, idir=0, max_iter=MAX_ITER)
    d = dict(LAST_DIAG)
    assert d["n_interfaces"] == n
    assert d["n_exact"] + d["n_hlld_fallback"] == n
    assert 0.0 <= d["frac_exact"] <= 1.0
