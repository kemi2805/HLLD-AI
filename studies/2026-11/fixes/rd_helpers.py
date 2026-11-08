"""Coplanar Alfven discontinuities (rotations through pi in the fluid frame)
from a state, found where the slow-family Hugoniot passes through rho
continuous at the Alfven speed.  In the lab frame such a rotation lands on
EITHER sign of B_t: production's reversal (planar5_b._reverse, the pinned
solve at the lab angle + pi) finds only the opposite-sign ones.

rd_scan(A, side, Bn)  -> list of (b, V, state) per lane: every rotation found
rd_refine(A, side, Bn, guess) -> the pinned Alfven solve at the guess's lab
                                 angle, seeded by the guess (tracks a known
                                 rotation as A moves)
"""
import os
import sys
import numpy as np
from rmhd.batched import slow_shock_b as SSB                      # noqa: E402
from rmhd.batched import fullcontact_b as FB                      # noqa: E402
import rmhd_ideal as RI                                           # noqa: E402


def make(gamma):
    sys.path.insert(0, os.environ.get("HLLD_SCRIPTS", "/Users/miler/Codes/HLLD/scripts"))
    import compound_wave as CW                                    # noqa: E402
    W = FB.make_waves(gamma)
    shock = SSB.make_solver(gamma)

    def _hug(A, side, Bn, b):
        sw = "LS" if side == "L" else "RS"
        k = b.size
        Ak = [np.repeat(c, k) for c in A]
        Bk = np.repeat(Bn, k)
        with CW.wide_window():
            V, S, e = shock(b, np.zeros(k), Ak, sw, Bk, accuracy=1e-12)
        drho = S[0] / Ak[0] - 1.0
        la = W["alfven_speed"](Ak, Bk, side)
        return V, S, np.asarray(e, bool), drho, la - V

    def rd_scan(A, side, Bn, ngrid=64, lo=1e-3, hi=30.0, nbis=50):
        """One lane (A as 7 length-1 arrays): every rotation on both branches."""
        bt = float(A[5][0])
        mag = np.exp(np.linspace(np.log(lo), np.log(hi), ngrid)) * max(abs(bt), 1e-6)
        found = []
        for sgn in (1.0, -1.0):
            b = sgn * np.sign(bt) * mag
            V, S, e, drho, gap = _hug(A, side, Bn, b)
            for t in range(ngrid - 1):
                if e[t] or e[t + 1] or not (np.isfinite(drho[t]) and np.isfinite(drho[t + 1])):
                    continue
                if np.sign(drho[t]) == np.sign(drho[t + 1]):
                    continue
                if sgn > 0 and (b[t] - bt) * (b[t + 1] - bt) <= 0:
                    continue                      # the trivial point b = B_t(A)
                if max(abs(gap[t]), abs(gap[t + 1])) > 1e-3:
                    continue
                x0, x1, d0 = b[t], b[t + 1], drho[t]
                for _ in range(nbis):
                    xm = 0.5 * (x0 + x1)
                    _, _, em, dm, _ = _hug(A, side, Bn, np.array([xm]))
                    if em[0] or not np.isfinite(dm[0]):
                        break
                    if np.sign(dm[0]) == np.sign(d0):
                        x0, d0 = xm, dm[0]
                    else:
                        x1 = xm
                Vm, Sm, em, dm, gm = _hug(A, side, Bn, np.array([0.5 * (x0 + x1)]))
                if not em[0]:
                    found.append((0.5 * (x0 + x1), float(Vm[0]), [c.copy() for c in Sm], float(dm[0]), float(gm[0])))
        return found

    def rd_refine(A, side, Bn, guess):
        Vs = W["alfven_speed"](A, Bn, side)
        out, err = W["alfven_checked"](Vs, A, guess, Bn)
        return out, Vs, err

    return dict(rd_scan=rd_scan, rd_refine=rd_refine, W=W)


def coplanar_rotations(ahead, Bn, Vs, gamma):
    """Both coplanar rotational discontinuities from ``ahead`` at speed ``Vs``
    in closed form (fullcontact_b.exact_rotation's construction): the line of
    four-velocities behind meets the unit hyperboloid in two points for each of
    the two lab angles psi_ahead + pi (the field reversed in the lab) and
    psi_ahead (the same lab sign -- one point is the identity, the other the
    rotation).  Returns ``{"rev": (state, ok), "same": (state, ok)}``."""
    from rmhd.batched import wave_speeds_b as WB
    _MINK = np.array([-1.0, 1.0, 1.0, 1.0])
    rho, P, vx, vy, vz, By, Bz = (np.asarray(c, dtype=float) for c in ahead)
    n = rho.shape[0]
    out = {}
    with np.errstate(all="ignore"):
        W = 1.0 / np.sqrt(np.maximum(1.0 - (vx * vx + vy * vy + vz * vz), 1e-300))
        vB = vx * Bn + vy * By + vz * Bz
        b = np.stack([W * vB, Bn / W + W * vB * vx, By / W + W * vB * vy, Bz / W + W * vB * vz], axis=1)
        b2 = (Bn * Bn + By * By + Bz * Bz) / (W * W) + vB * vB
        sE = np.sqrt(np.maximum(rho * WB.enthalpy(P - 0.5 * b2, rho, gamma) + b2, 0.0))
        u = np.stack([W, W * vx, W * vy, W * vz], axis=1)
        qp, qm = b + sE[:, None] * u, b - sE[:, None] * u
        s = np.where(np.abs(qp[:, 1] / qp[:, 0] - Vs) < np.abs(qm[:, 1] / qm[:, 0] - Vs), 1.0, -1.0)
        q = np.where(s[:, None] > 0, qp, qm)
        psi_a = np.arctan2(Bz, By)
        for label, psi in (("rev", psi_a + np.pi), ("same", psi_a)):
            c, sn = np.cos(psi), np.sin(psi)
            A = np.zeros((n, 3, 4))
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2], A[:, 0, 3] = -q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            A[:, 1, 0], A[:, 1, 1] = q[:, 1], -q[:, 0]
            A[:, 2, 0] = q[:, 2] * sn - q[:, 3] * c
            A[:, 2, 2], A[:, 2, 3] = -q[:, 0] * sn, q[:, 0] * c
            r = np.stack([-s * sE, Bn, np.zeros(n)], axis=1)
            At = np.transpose(A, (0, 2, 1))
            AAt = A @ At
            sing = ~np.isfinite(AAt).all(axis=(1, 2)) | (np.abs(np.linalg.det(np.where(np.isfinite(AAt), AAt, 0.0))) <= 1e-300)
            AAt[sing] = np.eye(3)
            up = (At @ np.linalg.solve(AAt, r[..., None]))[..., 0]
            d = np.stack([(-1.0) ** k * np.linalg.det(np.delete(A, k, axis=2)) for k in range(4)], axis=1)
            a2 = (_MINK * d * d).sum(axis=1)
            a1 = 2.0 * (_MINK * up * d).sum(axis=1)
            a0 = (_MINK * up * up).sum(axis=1) + 1.0
            root = np.sqrt(a1 * a1 - 4.0 * a2 * a0)
            best = [np.array(x, dtype=float, copy=True) for x in ahead]
            okall = np.zeros(n, dtype=bool)
            for t in ((-a1 - root) / (2.0 * a2), (-a1 + root) / (2.0 * a2)):
                U = up + t[:, None] * d
                vb = U[:, 1:] / U[:, :1]
                By_b = q[:, 2] * U[:, 0] - q[:, 0] * U[:, 2]
                Bz_b = q[:, 3] * U[:, 0] - q[:, 0] * U[:, 3]
                trivial = np.max(np.abs(U - u), axis=1) <= 1e-9 * np.max(np.abs(u), axis=1)
                ok = (~okall & ~sing & np.isfinite(U).all(axis=1) & (U[:, 0] >= 1.0)
                      & ((vb * vb).sum(axis=1) < 1.0) & (By_b * c + Bz_b * sn > 0.0) & ~trivial)
                for k, val in ((2, vb[:, 0]), (3, vb[:, 1]), (4, vb[:, 2]), (5, By_b), (6, Bz_b)):
                    best[k] = np.where(ok, val, best[k])
                okall |= ok
            out[label] = (best, okall)
    return out
