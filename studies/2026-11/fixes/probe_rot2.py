"""Inside exact_rotation for one state: both hyperboloid intersections and why
each is refused, for target angles psi_a + dphi."""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("RMHD_DIR", "/Users/miler/Codes/rmhd_final"))
np.seterr(all="ignore")
from rmhd.batched import planar5_b as P5, fullcontact_b as FB, wave_speeds_b as WB
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
p = np.load(POP, allow_pickle=True); g = float(p["gamma"])
S = P5.make_solver(g); W = S["waves"]
lane, PA, PD = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
UL, UR = p["UL"][lane], p["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0, alpha, _ = S["to_planar"](L, R, Bn)
A, _, _ = W["fast_wave"](np.array([PA]), L0, "LF", Bn)
_MINK = np.array([-1.0, 1.0, 1.0, 1.0])
for X, side in ((A, "L"),):
    Vs = W["alfven_speed"](X, Bn, side)
    rho, P, vx, vy, vz, By, Bz = (np.asarray(c, float) for c in X)
    Wl = 1/np.sqrt(1-(vx*vx+vy*vy+vz*vz)); vB = vx*Bn+vy*By+vz*Bz
    b = np.stack([Wl*vB, Bn/Wl+Wl*vB*vx, By/Wl+Wl*vB*vy, Bz/Wl+Wl*vB*vz], axis=1)
    b2 = (Bn*Bn+By*By+Bz*Bz)/(Wl*Wl)+vB*vB
    sE = np.sqrt(rho*WB.enthalpy(P-0.5*b2, rho, g)+b2)
    u = np.stack([Wl, Wl*vx, Wl*vy, Wl*vz], axis=1)
    qp, qm = b+sE[:, None]*u, b-sE[:, None]*u
    print("Vs %.7f  qp speed %.7f  qm speed %.7f" % (Vs[0], qp[0,1]/qp[0,0], qm[0,1]/qm[0,0]))
    for dphi in (0.3, 1.0, 2.0, 2.8, 3.0, np.pi):
        psi = np.arctan2(X[6], X[5]) + dphi
        for s in (1.0, -1.0):
            q = qp if s > 0 else qm
            c, sn = np.cos(psi), np.sin(psi)
            Am = np.zeros((1, 3, 4))
            Am[:, 0, 0], Am[:, 0, 1], Am[:, 0, 2], Am[:, 0, 3] = -q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            Am[:, 1, 0], Am[:, 1, 1] = q[:, 1], -q[:, 0]
            Am[:, 2, 0] = q[:, 2]*sn - q[:, 3]*c
            Am[:, 2, 2], Am[:, 2, 3] = -q[:, 0]*sn, q[:, 0]*c
            r = np.stack([-s*sE, Bn, np.zeros(1)], axis=1)
            At = np.transpose(Am, (0, 2, 1)); AAt = Am @ At
            up = (At @ np.linalg.solve(AAt, r[..., None]))[..., 0]
            d = np.stack([(-1.0)**k*np.linalg.det(np.delete(Am, k, axis=2)) for k in range(4)], axis=1)
            a2 = (_MINK*d*d).sum(1); a1 = 2*(_MINK*up*d).sum(1); a0 = (_MINK*up*up).sum(1)+1
            disc = a1*a1-4*a2*a0
            msg = "dphi %.2f s %+d disc %.2e" % (dphi, s, disc[0])
            if disc[0] >= 0:
                for t in ((-a1-np.sqrt(disc))/(2*a2), (-a1+np.sqrt(disc))/(2*a2)):
                    U = up+t[:, None]*d; vb = U[:, 1:]/U[:, :1]
                    Byb = q[:, 2]*U[:, 0]-q[:, 0]*U[:, 2]; Bzb = q[:, 3]*U[:, 0]-q[:, 0]*U[:, 3]
                    msg += " | u0 %.3f |v|2 %.4f Bt.dir %.4f By %.4f vt %.4f" % (U[0,0], (vb*vb).sum(), (Byb*c+Bzb*sn)[0], Byb[0], vb[0,1])
            print("  " + msg)
