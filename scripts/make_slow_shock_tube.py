"""Construct a 1D RMHD Riemann problem with a slow shock of chosen strength.

The usual way to get a test problem is to pick two states and hope the waves
that appear are the ones you wanted.  This goes the other way: it BUILDS the
solution with the exact solver's own wave routines and reads the right state
off the end of it, so the slow shock has exactly the strength asked for and
the whole solution is known by construction.

Left state -> weak fast wave -> rotational discontinuity -> slow SHOCK
           -> contact (density only) = right state.  The right-going family
is silent.  With the defaults the slow shock halves |B_t|, which at plasma
beta 0.04 compresses the gas 3.74x (the gamma = 5/3 limit is 4) and heats it
20.5x: almost all of the structure is in the one wave HLLD does not have.

One subtlety cost an afternoon.  The seven-wave unknowns include the contact
field angle psi_CD, and a relativistic slow shock does NOT deliver the angle
it is aimed at: with transverse velocity the downstream field is twisted in
the lab frame (4.5 deg here).  The solver reports the mismatch as the slow
wave's SLACK component and forces the requested angle, so a naive backward
construction leaves a residual of ~1e-2.  The fixed point psi_CD = delivered
angle is found by a secant iteration on slack_L(psi_CD) = 0 -- four steps to
1e-15.  A plain fixed-point update diverges (the slope is the wrong sign).

    python scripts/make_slow_shock_tube.py            # the defaults
    python scripts/make_slow_shock_tube.py --factor 3 --rot 0.3 --contact 1
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_RMHD = os.environ.get("RMHD_ROOT", "/Users/miler/Codes/rmhd_final")
sys.path.insert(0, _RMHD)
from batched.ray_scalar import claim_rmhd_namespace     # noqa: E402
claim_rmhd_namespace()
from eos import set_eos                                  # noqa: E402
set_eos("ideal")
from batched import fullcontact_b as FB                  # noqa: E402

G = 5.0 / 3.0
A = lambda *v: [np.array([float(x)]) for x in v]         # one-lane arrays


def gas_p(st, Bn):
    rho, P, vx, vy, vz, By, Bz = st
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    eta = Bn * vx + By * vy + Bz * vz
    return P - 0.5 * ((Bn * Bn + By * By + Bz * Bz) / W2 + eta * eta)


def build(a):
    f6 = FB.make_solver(G)["fullfuncv6"]
    Bn = a.Bn
    vx, vy, vz, By, Bz = a.vx, a.vy, a.vz, a.Bt, 0.0
    W2 = 1.0 / (1.0 - (vx * vx + vy * vy + vz * vz))
    b2 = (Bn * Bn + By * By + Bz * Bz) / W2 + (Bn * vx + By * vy + Bz * vz) ** 2
    L = [a.rho, a.p + 0.5 * b2, vx, vy, vz, By, Bz]
    lnB = np.log(np.hypot(By, Bz) / a.factor)
    psi_L = np.arctan2(Bz, By)

    def slack(psi_t):
        unk = A(np.log(L[1] * a.fast), lnB, psi_t, np.log(L[1]), a.rot, 0.0)
        fv, Z, VL, VR, e = f6(A(*L), A(*L), unk, np.array([Bn]))
        if bool(e[0]):
            raise RuntimeError("structure not constructible at psi_t=%g" % psi_t)
        return float(fv[4][0])

    p0, p1 = psi_L + a.rot, psi_L + a.rot + 0.02
    s0, s1 = slack(p0), slack(p1)
    for _ in range(40):
        p2 = p1 - s1 * (p1 - p0) / (s1 - s0)
        s2 = slack(p2)
        p0, s0, p1, s1 = p1, s1, p2, s2
        if abs(s2) < 1e-13:
            break
    psi_t = p1
    unk = A(np.log(L[1] * a.fast), lnB, psi_t, np.log(L[1]), a.rot, 0.0)
    fv, Z, VL, VR, e = f6(A(*L), A(*L), unk, np.array([Bn]))
    zs = [[float(z[0]) for z in zz] for zz in Z]
    R4 = zs[2]
    R = list(R4)
    R[0] *= a.contact
    # verify on the actual problem
    unk_t = A(np.log(L[1] * a.fast), np.log(np.hypot(R[5], R[6])),
              np.arctan2(R[6], R[5]), np.log(R[1]), a.rot, 0.0)
    fv, Z, VL, VR, e = f6(A(*L), A(*R), unk_t, np.array([Bn]))
    zs = [[float(z[0]) for z in zz] for zz in Z]
    VLs = [float(v[0]) for v in VL]
    VRs = [float(v[0]) for v in VR]
    cd = 0.5 * (zs[2][2] + zs[3][2])
    speeds = np.array([VLs[0], VLs[1], VLs[2], cd, VRs[2], VRs[1], VRs[0]])
    return dict(L=L, R=R, Bn=Bn, unk=np.array([float(c[0]) for c in unk_t]),
                resid=float(np.max(np.abs(fv))), speeds=speeds, zones=zs,
                psi_t=psi_t, twist_deg=np.degrees(psi_t - psi_L - a.rot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=0.1, help="left GAS pressure")
    ap.add_argument("--vx", type=float, default=0.1)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vz", type=float, default=0.0)
    ap.add_argument("--Bn", type=float, default=1.0)
    ap.add_argument("--Bt", type=float, default=2.0, help="left |B_t| (along y)")
    ap.add_argument("--fast", type=float, default=1.15, help="P_tot ratio across the fast wave")
    ap.add_argument("--rot", type=float, default=0.5, help="rotation angle phi_L [rad]")
    ap.add_argument("--factor", type=float, default=2.0, help="|B_t| DROP across the slow shock")
    ap.add_argument("--contact", type=float, default=2.0, help="density ratio across the contact")
    ap.add_argument("--yaml", default="configs/slow_shock_tube.yaml")
    ap.add_argument("--npz", default="results/slow_shock_design.npz")
    a = ap.parse_args()
    d = build(a)
    L, R, Bn = d["L"], d["R"], d["Bn"]
    zs, sp = d["zones"], d["speeds"]
    print("residual of the constructed solution: %.2e" % d["resid"])
    print("speeds [LF LA LS CD RS RA RF] = %s   ordered: %s"
          % (np.round(sp, 4), bool(np.all(np.diff(sp) >= -1e-9))))
    print("fast wave:   rho %.3f -> %.3f   p_gas %.3f -> %.3f"
          % (L[0], zs[0][0], gas_p(L, Bn), gas_p(zs[0], Bn)))
    print("Alfven:      field angle %.1f -> %.1f deg"
          % (np.degrees(np.arctan2(zs[0][6], zs[0][5])), np.degrees(np.arctan2(zs[1][6], zs[1][5]))))
    print("SLOW SHOCK:  rho %.3f -> %.3f (x%.2f)   p_gas %.3f -> %.3f (x%.1f)   |Bt| %.3f -> %.3f   "
          "vy %.3f -> %.3f   lab-frame twist %.2f deg"
          % (zs[1][0], zs[2][0], zs[2][0] / zs[1][0], gas_p(zs[1], Bn), gas_p(zs[2], Bn),
             gas_p(zs[2], Bn) / gas_p(zs[1], Bn), np.hypot(zs[1][5], zs[1][6]),
             np.hypot(zs[2][5], zs[2][6]), zs[1][3], zs[2][3], d["twist_deg"]))
    print("contact:     rho %.3f -> %.3f" % (zs[2][0], R[0]))
    chi = abs(L[5] * R[6] - L[6] * R[5]) / np.hypot(L[5], L[6]) / np.hypot(R[5], R[6])
    print("coplanarity chi = %.3f" % chi)

    fmt = lambda st: "\n".join("    %-4s %.12g" % (k + ":", v) for k, v in
                               zip(("rho", "p", "vx", "vy", "vz", "Bx", "By", "Bz"),
                                   (st[0], gas_p(st, Bn), st[2], st[3], st[4], Bn, st[5], st[6])))
    y = """# ============================================================
#  1D SR MHD -- Slow-shock tube.  GENERATED by scripts/make_slow_shock_tube.py
#
#  Gamma = 5/3.  Built backward from the wave curves so the exact solution
#  is known by construction: a weak left fast wave (P_tot x%.2f), a %.2f rad
#  rotational discontinuity, a slow SHOCK dropping |B_t| by %.1fx, and a
#  contact with density ratio %.1f.  The right family is silent.
#  Constructed-solution residual %.1e; chi = %.3f (non-coplanar).
# ============================================================

grid:
  xmin:   0.0
  xmax:   1.0
  ncells: 400
  ng:     2

eos:
  K:        0.0
  gamma:    1.6666666
  gamma_th: 1.6666666

run:
  problem:  generic
  t_end:    0.4
  cfl:      0.4
  atmo_rho: 1.0e-10
  solver:   hlld
  device:   cpu

  primL:
%s
  primR:
%s
  x_interface: 0.5

output:
  dir:           output_slow_shock
  every_n_steps: 1000
""" % (a.fast, a.rot, a.factor, a.contact, d["resid"], chi, fmt(L), fmt(R))
    os.makedirs(os.path.dirname(a.yaml) or ".", exist_ok=True)
    open(a.yaml, "w").write(y)
    os.makedirs(os.path.dirname(a.npz) or ".", exist_ok=True)
    np.savez(a.npz, L=np.array(L), R=np.array(R), Bn=Bn, unk=d["unk"], speeds=sp,
             gamma=G, psi_t=d["psi_t"], resid=d["resid"])
    print("wrote %s and %s" % (a.yaml, a.npz))


if __name__ == "__main__":
    main()
