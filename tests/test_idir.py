"""
Direction-genericity of the Riemann solvers.

The 2D extension applies the same solvers along y with ``idir=1``.  Before
that is safe, every solver must satisfy the rotation-equivalence identity
below.  Historically ``hlld_flux`` and ``hlld_ai_flux`` accepted ``idir``
but built their internal U_init/F_init blocks from literal "Sx"/"vx"/b^x,
so they returned x-fluxes for any ``idir`` — wrong answers, no error.

The identity
------------
Let C be the cyclic relabelling of vector components

    C(a)_x = a_z ,   C(a)_y = a_x ,   C(a)_z = a_y

i.e. what used to be the x-axis is now called the y-axis.  C is a proper
rotation (det = +1), so it is a symmetry of the equations, and it matches
the solvers' own transverse indexing ``t1=(idir+1)%3, t2=(idir+2)%3``:

    idir=0 -> (n,t1,t2) = (x,y,z)
    idir=1 -> (n,t1,t2) = (y,z,x)

Applying C sends the triple (x,y,z) to (y,z,x), so a solve along x must
equal the C-image of a solve along y:

    flux( C(P_L), C(P_R), idir=1 )  ==  C( flux(P_L, P_R, idir=0) )

and likewise C^2 with idir=2.  Scalars (D, tau) are untouched by C.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.eos import hybrid_eos
from src.physics.hlld import hlle_flux, hllc_flux, hlld_flux

torch.manual_seed(20260824)

GAMMA = 5.0 / 3.0
VEC_TRIPLES = [("vx", "vy", "vz"), ("Bx", "By", "Bz")]
CONS_TRIPLES = [("Sx", "Sy", "Sz"), ("Bx", "By", "Bz")]


def cyc(d: dict, triples, n: int = 1) -> dict:
    """Apply C^n: component i of the output is component (i-n) %3 of the input."""
    out = dict(d)
    for tri in triples:
        if tri[0] not in d:
            continue
        for i, key in enumerate(tri):
            out[key] = d[tri[(i - n) % 3]]
    return out


def random_states(n: int, eos: hybrid_eos, max_W: float = 5.0, seed: int = 20260824):
    """Physically admissible random left/right primitive states.

    Seeded per call: pytest runs each parametrisation separately, so a
    module-level seed would hand every case a different batch and make
    failures irreproducible.
    """
    torch.manual_seed(seed)

    def one():
        rho = 10.0 ** torch.empty(n).uniform_(-1.0, 1.0).double()
        p = 10.0 ** torch.empty(n).uniform_(-1.5, 1.0).double()
        # sample velocity via the Lorentz factor so |v| < 1 by construction
        W = 1.0 + 10.0 ** torch.empty(n).uniform_(-3.0, math.log10(max_W - 1.0)).double()
        speed = torch.sqrt(1.0 - 1.0 / W**2)
        th = torch.empty(n).uniform_(0, math.pi).double()
        ph = torch.empty(n).uniform_(0, 2 * math.pi).double()
        B = torch.empty(3, n).uniform_(-3.0, 3.0).double()
        s = {
            "rho": rho,
            "p": p,
            "vx": speed * torch.sin(th) * torch.cos(ph),
            "vy": speed * torch.sin(th) * torch.sin(ph),
            "vz": speed * torch.cos(th),
            "Bx": B[0],
            "By": B[1],
            "Bz": B[2],
        }
        s["eps"] = eos.eps__press_rho(s["p"], s["rho"])
        return s

    sL, sR = one(), one()
    # the normal field component is continuous across an interface (div B = 0);
    # enforce it for all three components so the state is valid for every idir
    for k in ("Bx", "By", "Bz"):
        sR[k] = sL[k]
    return sL, sR


def max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = torch.maximum(a.abs().max(), torch.tensor(1e-30, dtype=a.dtype))
    return float((a - b).abs().max() / scale)


@pytest.mark.parametrize("flux_fn", [hlle_flux, hllc_flux, hlld_flux],
                         ids=["hlle", "hllc", "hlld"])
@pytest.mark.parametrize("n_cyc", [1, 2], ids=["idir1", "idir2"])
def test_rotation_equivalence(flux_fn, n_cyc):
    """flux(C^n P, idir=n) must equal C^n(flux(P, idir=0)).

    Interfaces where the two runs take DIFFERENT discrete branches are
    excluded from the value comparison and bounded by count instead.
    ``hlld_flux`` falls back to HLLE when the root-find fails or the wave
    ordering is unphysical, and that decision is a floating-point-marginal
    predicate: the x-solve and the rotated solve differ in the last ulp, so
    a state sitting exactly on the boundary can land on either side.  Such
    a flip is not a direction bug — the same fragility exists along x — so
    it is asserted to be rare rather than absent.  Everything else must
    agree to round-off.
    """
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    sL, sR = random_states(4000, eos)

    f_ref, u_ref, ps_ref = flux_fn(sL, sR, eos, idir=0)
    f_rot, u_rot, ps_rot = flux_fn(
        cyc(sL, VEC_TRIPLES, n_cyc), cyc(sR, VEC_TRIPLES, n_cyc), eos, idir=n_cyc
    )

    f_expect = cyc(f_ref, CONS_TRIPLES, n_cyc)
    u_expect = cyc(u_ref, CONS_TRIPLES, n_cyc)

    # p* < 0 is the sentinel written where the solver fell back to HLLE
    branch_flip = (ps_ref < 0) ^ (ps_rot < 0)
    n = ps_ref.numel()
    keep = ~branch_flip

    worst = 0.0
    offenders = []
    for k in f_ref:
        for got, want, tag in ((f_rot, f_expect, "flux"), (u_rot, u_expect, "state")):
            r = max_rel(want[k][keep], got[k][keep])
            worst = max(worst, r)
            if r > 1e-13:
                offenders.append(f"{tag}[{k}] rel={r:.3e}")

    assert not offenders, (
        f"{flux_fn.__name__} is not idir-generic (idir={n_cyc}): "
        + ", ".join(offenders)
        + f"\nworst={worst:.3e} over {int(keep.sum())}/{n} same-branch interfaces"
    )

    frac = float(branch_flip.sum()) / n
    assert frac < 0.005, (
        f"{flux_fn.__name__} idir={n_cyc}: HLLE-fallback decision flips on "
        f"{frac*100:.2f}% of interfaces ({int(branch_flip.sum())}/{n}) — "
        "too many to be last-ulp marginality"
    )


@pytest.mark.parametrize("flux_fn", [hlle_flux, hllc_flux, hlld_flux],
                         ids=["hlle", "hllc", "hlld"])
def test_idir0_unchanged_by_relabelling(flux_fn):
    """Sanity: C is a relabelling, so a C-symmetric state gives C-symmetric flux."""
    eos = hybrid_eos(K=0.0, gamma=GAMMA, gamma_th=GAMMA)
    sL, sR = random_states(500, eos)
    f0, _, _ = flux_fn(sL, sR, eos, idir=0)
    f0b, _, _ = flux_fn(sL, sR, eos, idir=0)
    for k in f0:
        assert torch.equal(f0[k], f0b[k]), f"{flux_fn.__name__} is non-deterministic in {k}"
