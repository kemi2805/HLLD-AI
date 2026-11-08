"""A kind-S (signed field) planar4 answer of one lane: every wave's type,
speed against the Alfven speeds either side, entropy jump; the xi = 0 flux
through production's acceptance (hlld_fix _planar4_accept) vs HLLD."""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.environ.get("HLLD_DIR", "/Users/miler/Codes/HLLD"))
np.seterr(all="ignore")
from rmhd.eos import set_eos
set_eos("ideal")
from rmhd.batched import planar4_b as P4B, wave_speeds_b as WB
from src.physics import exact_flux as EF
from src.physics.eos import hybrid_eos
from src.physics.hlld import compute_srmhd_fluxes
POP = "/Users/miler/Codes/claude_scratch/r2d_retrain/anatomy/data/ledger/population.npz"
pop = np.load(POP, allow_pickle=True); g = float(pop["gamma"])
off = np.load("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/replay_fix3/replay_off.npz")
P = EF._batched_parts(g); S4 = P["planar4"]; W = S4["waves"]
lane = int(sys.argv[1]); kd = tuple(int(c) for c in sys.argv[2]); dl = 1e-4
UL, UR = pop["UL"][lane], pop["UR"][lane]
Bn = np.array([UL[7]])
L = [np.array([UL[j]]) for j in range(7)]; R = [np.array([UR[j]]) for j in range(7)]
L0, R0 = S4["to_planar"](L, R, Bn)[:2]
ps = abs(off["pstar"][lane])
A, _, _ = W["fast_wave"](np.array([ps]), L0, "LF", Bn)
D, _, _ = W["fast_wave"](np.array([ps]), R0, "RF", Bn)
st0 = lambda k, X: np.log(X[1] * (1 - dl)) if k == 1 else X[5] * (1 + dl)
u0 = [np.log(np.array([ps])), st0(kd[0], A), st0(kd[1], D), np.log(np.array([ps]))]
r = S4["solve"](L, R, Bn, kinds=np.array(kd, np.int8).reshape(2, 1), retry=False, unk_init=u0, central=True,
                planar_tol=EF._PLANAR_TOL, verify_tol=EF._PLANAR5_VERIFY)
print("lane %d kinds %s: converged %s reason %d cert %.1e ordered %s" % (lane, kd, r["converged"][0], r["reason"][0], r["cert_resid"][0], r["ordered"][0]))
if not r["converged"][0]:
    sys.exit()
good, reason, st, reg, z6, VsLv, VsRv = EF._planar4_accept(P, g, r, L, R, Bn)
print("  production acceptance (ray, physical):", good[0], "reason", reason[0], "region", reg[0] if reg is not None else None)
def K(X):
    rho, Pt, vx, vy, vz, By, Bz = X
    v2 = vx*vx + vy*vy + vz*vz; W2 = 1/(1 - v2); vB = vx*Bn + vy*By + vz*Bz
    return (Pt - 0.5*((Bn**2 + By**2 + Bz**2)/W2 + vB**2)) / X[0]**g
Az, Bz_, Cz, Dz = r["zones"]
names = ("LF", "LS", "RS", "RF")
for row, (ah, bh, V, sw) in enumerate(((L, Az, r["speeds"][0], "L"), (Az, Bz_, r["speeds"][1], "L"),
                                       (Dz, Cz, r["speeds"][3], "R"), (R, Dz, r["speeds"][4], "R"))):
    ea, laa, raa, _ = WB.xi_all(*ah, Bn, g); eb, lab, rab, _ = WB.xi_all(*bh, Bn, g)
    al_a = (laa if sw == "L" else raa)[0]; al_b = (lab if sw == "L" else rab)[0]
    print("  %s %s V %.6f  Alfven ahead %.6f behind %.6f  Bt %+.4f -> %+.4f  dS %+.1e" % (
        names[row], "shock" if r["shocks"][row, 0] else "fan  ", V[0], al_a, al_b, ah[5][0], bh[5][0], (K(bh)/K(ah))[0] - 1))
if good[0]:
    eos = hybrid_eos(K=0.0, gamma=g, gamma_th=g)
    rho, Pt, vx, vy, vz, By, Bz = (float(c[0]) for c in st)
    W2 = 1/(1 - vx*vx - vy*vy - vz*vz); vB = vx*Bn[0] + vy*By + vz*Bz
    s_ = {"rho": rho, "p": Pt - 0.5*((Bn[0]**2 + By**2 + Bz**2)/W2 + vB**2), "vx": vx, "vy": vy, "vz": vz, "Bx": Bn[0], "By": By, "Bz": Bz}
    s_ = {k: torch.tensor([v], dtype=torch.float64) for k, v in s_.items()}
    s_["eps"] = eos.eps__press_rho(s_["p"], s_["rho"])
    _, _, F, _, _, _ = compute_srmhd_fluxes(s_, s_, eos, idir=0)
    KEYS = ("D", "Sx", "Sy", "Sz", "tau", "By", "Bz")
    Fex = np.array([float(F[k][0]) for k in KEYS]); Fh = np.array([off["F_" + k][lane] for k in KEYS])
    np.save("/Users/miler/Codes/claude_scratch/r2d_retrain/fixes/sflux_%d.npy" % lane, np.stack([Fex, Fh]))
    print("  |F_exact - F_HLLD| / |F| = %.1e" % (np.abs(Fex - Fh).max() / np.abs(Fh).max()))
