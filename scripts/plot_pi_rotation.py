"""A rotor interface whose exact solution contains a pi ROTATION.

These are the ~2.7% the five-wave planar family cannot represent: a
rotational discontinuity of zero width but non-zero strength, which reverses
the tangential field while leaving density, pressure and |B_t| untouched.
`planar5_b.verify_full` is what refuses to fake them.

They are also crowded: the Alfven, slow and contact waves sit inside a window
~1e-2 wide, so the figure needs both scales.
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rmhd.eos import set_eos
set_eos("ideal")
from rmhd.batched import planar5_b as P5
from rmhd.batched import ray_b as RAY
from rmhd.batched import rarefaction_b as RB
from rmhd.batched import wave_speeds_b as WB

np.seterr(all="ignore")
G = 5.0 / 3.0
H = "/Users/miler/Codes/HLLD/results/rotor_64_exact/harvest"
ROW = int(os.environ.get("ROW", "3928"))
_IDX = {"LF": 0, "LS": 1, "RS": 2, "RF": 3}
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def xi_fn(state, switch, Bn, g=G):
    eig, _, _, ok = WB.xi_all(*state, Bn, g)
    return np.where(ok[:, _IDX[switch]], eig[:, _IDX[switch]], np.nan)


fan_p, fan_n = RB.make_integrators(G, lambda s, sw, B, g: xi_fn(s, sw, B, g))

fs = sorted(glob.glob("%s/solved_*.npz" % H))[:8]
Z = np.concatenate([np.load(f)["zones"] for f in fs]).astype(float)[ROW]
SP = np.concatenate([np.load(f)["speeds"] for f in fs]).astype(float)[ROW]
UL = np.concatenate([np.load(f)["U_L"] for f in fs]).astype(float)[ROW]
Bn = np.array([UL[7]])

left = [np.array([Z[0, j]]) for j in range(7)]
right = [np.array([Z[7, j]]) for j in range(7)]
_, _, alpha, _ = P5.to_planar(left, right, Bn)
ca, sa = float(np.cos(alpha[0])), float(np.sin(alpha[0]))


def rot(u):
    o = list(u)
    o[3], o[4] = ca * u[3] - sa * u[4], sa * u[3] + ca * u[4]
    o[5], o[6] = ca * u[5] - sa * u[6], sa * u[5] + ca * u[6]
    return o


ZR = np.array([rot(Z[k]) for k in range(8)])
left = [np.array([ZR[0, j]]) for j in range(7)]
right = [np.array([ZR[7, j]]) for j in range(7)]
zones6 = [[np.array([ZR[k, j]]) for j in range(7)] for k in range(1, 7)]
VsLv = [np.array([SP[0]]), np.array([SP[1]]), np.array([SP[2]])]
VsRv = [np.array([SP[6]]), np.array([SP[5]]), np.array([SP[4]])]

psi = lambda k: np.arctan2(ZR[k, 6], ZR[k, 5])
print("left rotation  R2->R3: %.6f rad = %.4f pi" % (wrap(psi(2) - psi(1)),
                                                     wrap(psi(2) - psi(1)) / np.pi))
print("right rotation R6->R7: %.6f rad = %.4f pi" % (wrap(psi(6) - psi(5)),
                                                     wrap(psi(6) - psi(5)) / np.pi))
print("|Bt| across the left rotation: %.6f -> %.6f"
      % (np.hypot(ZR[1, 5], ZR[1, 6]), np.hypot(ZR[2, 5], ZR[2, 6])))
print("rho  across the left rotation: %.6f -> %.6f" % (ZR[1, 0], ZR[2, 0]))
print("ptot across the left rotation: %.6f -> %.6f" % (ZR[1, 1], ZR[2, 1]))


def sample(lo, hi, n):
    xi = np.linspace(lo, hi, n)
    rep = lambda c: np.repeat(c[0], n)
    st, reg, err = RAY.state_at_xi(
        [rep(c) for c in left], [rep(c) for c in right],
        [[rep(c) for c in z] for z in zones6],
        [rep(v) for v in VsLv], [rep(v) for v in VsRv],
        np.repeat(Bn[0], n), G, xi_fn, fan_p, fan_n, xi_target=xi)
    return xi, st, reg, int(err.sum())


full = sample(SP[0] - 0.12, SP[6] + 0.12, 4000)
zlo, zhi = SP[1] - 0.0013, SP[2] + 0.0013   # the LEFT rotation, tight
zoom = sample(zlo, zhi, 4000)
print("ray failures: full %d, zoom %d" % (full[3], zoom[3]))

INK, ACC, MUT, FANC = "#1a1a1a", "#1f4e79", "#8a8a8a", "#dce7f2"
ROT = "#b3541e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c8c8c8", "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white"})
names = ["LF", "LA", "LS", "CD", "RS", "RA", "RF"]
FANR = (1, 4, 7, 10)


def series(st):
    rho, Ptot, vx, vy, vz, By, Bz = st
    v2 = vx ** 2 + vy ** 2 + vz ** 2
    Wl = 1.0 / np.sqrt(np.maximum(1 - v2, 1e-16))
    eta = Bn[0] * vx + By * vy + Bz * vz
    b2 = (Bn[0] ** 2 + By ** 2 + Bz ** 2) / Wl ** 2 + eta ** 2
    return {"rho": rho, "pgas": Ptot - 0.5 * b2, "Bt": np.hypot(By, Bz),
            "W": Wl, "b": np.sqrt(np.maximum(b2, 0.0)),
            "psi": np.unwrap(np.arctan2(Bz, By)) / np.pi}


rows = [("rho", r"$\rho$", "rest-mass density  —  CONTINUOUS across LA, RA"),
        ("W", r"$W$", "Lorentz factor  —  jumps"),
        ("Bt", r"$|\mathbf{B}^t|$", "lab-frame tangential field  —  jumps"),
        ("b", r"$\sqrt{b^2}$", "COMOVING field magnitude  —  CONTINUOUS"),
        ("psi", r"$\psi/\pi$", "field direction  —  steps by exactly $\pi$")]
fig, axes = plt.subplots(5, 2, figsize=(11.6, 11.8),
                         gridspec_kw={"width_ratios": [1.25, 1]})
for col, (xi, st, reg, _) in enumerate((full, zoom)):
    S = series(st)
    isf = np.isin(reg, FANR)
    bands, i = [], 0
    while i < len(isf):
        if isf[i]:
            j = i
            while j + 1 < len(isf) and isf[j + 1]:
                j += 1
            bands.append((xi[i], xi[j])); i = j + 1
        else:
            i += 1
    for r, (key, sym, title) in enumerate(rows):
        ax = axes[r, col]
        for a, b in bands:
            ax.axvspan(a, b, color=FANC, lw=0, zorder=0)
        if col == 0:
            ax.axvspan(zlo, zhi, color="#f6e7dc", lw=0, zorder=0)
        for s, nm in zip(SP, names):
            isrot = nm in ("LA", "RA")
            ax.axvline(s, color=ROT if isrot else MUT, lw=1.1 if isrot else 0.7,
                       ls=(0, (4, 3)), zorder=1)
        ax.plot(xi, S[key], color=ACC, lw=1.8, zorder=3)
        ax.set_xlim(xi[0], xi[-1])
        if col == 0:
            ax.set_ylabel(sym, fontsize=11)
        ax.set_title(title if col == 0 else "zoom: the left rotation (LA)",
                     fontsize=9, color=MUT,
                     loc="left", pad=16 if r == 0 else 5)
        ax.grid(axis="y", color="#ededed", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        y = S[key]
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 0.14 * (hi - lo + 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        if r == 0:
            yl = ax.get_ylim()[1]
            for n, (s, nm) in enumerate(zip(SP, names)):
                if col == 1 and not (xi[0] <= s <= xi[-1]):
                    continue
                ax.annotate(nm, xy=(s, yl), xytext=(0, 3 + 10 * (n % 2)),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=7.6, annotation_clip=False,
                            color=ROT if nm in ("LA", "RA") else MUT)
        if r == len(rows) - 1:
            ax.set_xlabel(r"$\xi = x/t$")

fig.suptitle("Exact seven-wave RMHD solution containing two $\\pi$ ROTATIONS "
             "— the case the five-wave family cannot represent",
             fontsize=11.5, x=0.010, ha="left", y=0.996)
fig.text(0.010, 0.966,
         "LA and RA (orange) are the rotational discontinuities.  The RELATIVISTIC "
         "rule differs from the Newtonian one: across each, $\\rho$, $p_{\\rm tot}$, "
         "$p_{\\rm gas}$ and the COMOVING field $b^2$ are continuous to machine\n"
         "precision, while the lab-frame $|\\mathbf{B}^t|$ changes ($0.9392 \\to 0.7687$) "
         "because $W$ jumps ($1.856 \\to 2.483$).  Only the direction is required to "
         "reverse, and it does so by exactly $\\pi$.\n"
         "The right column magnifies LA (orange band, left column).  RA behaves "
         "identically: $b^2$ continuous at $0.570920$, $|\\mathbf{B}^t|$ "
         "$0.6617 \\to 0.8642$, $W$ $2.572 \\to 3.504$.\n"
         "Shaded blue = rarefaction fans.  $B^n = %.4f$." % Bn[0],
         fontsize=8.2, color=MUT, ha="left", va="top")
fig.tight_layout(rect=[0, 0, 1, 0.930])
out = "/Users/miler/Codes/HLLD/figs/pi_rotation_profile.png"
fig.savefig(out, dpi=165)
print("wrote", out)
