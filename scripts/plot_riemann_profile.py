"""Plot the exact solution of a planar Riemann problem the solver rescued.

The interface is a REAL one, taken from the unsolved shard of a 64^2 rotor
run -- i.e. one the seven-wave solver failed on and the five-wave planar
solver resolved.  The profile is the exact self-similar solution sampled with
the same ray machinery the flux uses, so what is plotted is what the scheme
actually sees, not a reconstruction of it.
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
from rmhd.batched import fullcontact_b as FB
from rmhd.batched import planar5_b as P5
from rmhd.batched import ray_b as RAY
from rmhd.batched import rarefaction_b as RB
from rmhd.batched import wave_speeds_b as WB

np.seterr(all="ignore")
G = 5.0 / 3.0
S = P5.make_solver(G)
W = FB.make_waves(G)
H = os.environ.get("HARVEST", "results/rotor_64_exact/harvest")
OUT_NPZ = os.environ.get("PROFILE_NPZ", "figs/riemann_profile.npz")
os.makedirs("figs", exist_ok=True)
_IDX = {"LF": 0, "LS": 1, "RS": 2, "RF": 3}


def xi_fn(state, switch, Bn, g=G):
    eig, _, _, ok = WB.xi_all(*state, Bn, g)
    k = _IDX[switch]
    return np.where(ok[:, k], eig[:, k], np.nan)


fan_p, fan_n = RB.make_integrators(G, lambda s, sw, B, g: xi_fn(s, sw, B, g))

# ── pick an interface the planar solver rescues, with a visible structure ──
fs = sorted(glob.glob("%s/unsolved_*.npz" % H))[:2]
La = np.concatenate([np.load(f)["U_L"] for f in fs])[:6000].astype(float)
Ra = np.concatenate([np.load(f)["U_R"] for f in fs])[:6000].astype(float)
left = [La[:, j].copy() for j in range(7)]
right = [Ra[:, j].copy() for j in range(7)]
Bn = La[:, 7].copy()
r = S["solve"](left, right, Bn, accuracy=1e-10, max_iter=60)
ok = np.flatnonzero(r["converged"])
print("planar solver rescues %d of %d" % (len(ok), len(Bn)))

L, R, alpha, _ = P5.to_planar(left, right, Bn)
unk = [c.copy() for c in r["unk"]]
fvec, Z4, Vs, bad, slack, mid = S["structure"](L, R, unk, Bn)
spread = np.max(np.stack(Vs), axis=0) - np.min(np.stack(Vs), axis=0)
jump = np.abs(L[0] - R[0]) / np.maximum(L[0], R[0])
score = np.where(r["converged"] & ~bad, spread * (0.2 + jump), -1.0)
i = int(np.argmax(score))
print("chosen lane %d: wave-speed spread %.3f, density contrast %.3f"
      % (i, spread[i], jump[i]))

one = lambda c: np.array([c[i]])
Lo = [one(c) for c in L]; Ro = [one(c) for c in R]; Bo = np.array([Bn[i]])
uo = [one(c) for c in unk]
fv, Z, V, bd, sl, (A2, D2) = S["structure"](Lo, Ro, uo, Bo)
A, B_, C_, D = Z
# the seven-wave zone layout: both rotations have zero strength
zones7 = [A, A2, B_, C_, D2, D]
VsLv = [V[0], W["alfven_speed"](A, Bo, "L"), V[1]]
VsRv = [V[4], W["alfven_speed"](D, Bo, "R"), V[3]]

# verification: is this answer exact for the FULL seven-wave system?
full = r["full_resid"][i] if "full_resid" in r else np.nan
print("full seven-wave residual at this answer: %.3e" % full)

# ── sample the self-similar solution ──────────────────────────────────────
lo = min(V[k][0] for k in range(5)); hi = max(V[k][0] for k in range(5))
pad = 0.25 * max(hi - lo, 1e-3)
NX = 3000
xi = np.linspace(lo - pad, hi + pad, NX)
rep = lambda c: np.repeat(c[0], NX)
st, reg, err = RAY.state_at_xi([rep(c) for c in Lo], [rep(c) for c in Ro],
                               [[rep(c) for c in z] for z in zones7],
                               [rep([v]) if np.ndim(v) == 0 else np.repeat(v[0], NX)
                                for v in VsLv],
                               [rep([v]) if np.ndim(v) == 0 else np.repeat(v[0], NX)
                                for v in VsRv],
                               np.repeat(Bo[0], NX), G, xi_fn, fan_p, fan_n,
                               xi_target=xi)
print("ray failures: %d of %d" % (int(err.sum()), NX))

rho, Ptot, vx, vy, vz, By, Bz = st
bn = Bo[0]
v2 = vx ** 2 + vy ** 2 + vz ** 2
Wl = 1.0 / np.sqrt(np.maximum(1.0 - v2, 1e-16))
eta = bn * vx + By * vy + Bz * vz
b2 = (bn ** 2 + By ** 2 + Bz ** 2) / Wl ** 2 + eta ** 2
pgas = Ptot - 0.5 * b2

kinds = RAY.wave_kinds(Lo, Ro, zones7)
print("wave_kinds ->", kinds)
np.savez(OUT_NPZ,
         xi=xi, rho=rho, Ptot=Ptot, pgas=pgas, vx=vx, vy=vy, By=By, W=Wl,
         reg=reg, kinds=np.array([bool(np.ravel(kinds[k])[0])
                                  for k in ("LF", "LS", "RS", "RF")]),
         speeds=np.array([V[k][0] for k in range(5)]), Bn=bn,
         L=np.array([Lo[j][0] for j in range(7)]),
         R=np.array([Ro[j][0] for j in range(7)]), full=full)
print("saved", OUT_NPZ)

# ── figure ──────────────────────────────────────────────────────────────
# (continues in the same process)
z = np.load(OUT_NPZ)
xi, sp, reg = z["xi"], z["speeds"], z["reg"]
L, R, Bn = z["L"], z["R"], float(z["Bn"])
FAN = (1, 4, 7, 10)                       # ray_b.FAN_REGIONS
# saved in the order LF, LS, RS, RF; True = shock
kLF, kLS, kRS, kRF = [bool(b) for b in z["kinds"]]
kind = {"LF": kLF, "LS": kLS, "CD": None, "RS": kRS, "RF": kRF}

INK, ACC, MUT, FAN_C = "#1a1a1a", "#1f4e79", "#8a8a8a", "#dce7f2"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c8c8c8", "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white"})

panels = [("rho",  r"$\rho$",         "rest-mass density"),
          ("pgas", r"$p_{\rm gas}$",  "gas pressure"),
          ("Ptot", r"$p_{\rm tot}$",  "total pressure"),
          ("vx",   r"$v^x$",          "normal velocity"),
          ("vy",   r"$v^t$",          "tangential velocity"),
          ("By",   r"$B^t$",          "tangential field")]
names = ["LF", "LS", "CD", "RS", "RF"]

# contiguous runs of fan cells -> shaded bands
isfan = np.isin(reg, FAN)
bands, i = [], 0
while i < len(isfan):
    if isfan[i]:
        j = i
        while j + 1 < len(isfan) and isfan[j + 1]:
            j += 1
        bands.append((xi[i], xi[j]))
        i = j + 1
    else:
        i += 1

fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.1), sharex=True)
for k, (ax, (key, sym, title)) in enumerate(zip(axes.ravel(), panels)):
    y = z[key]
    for a, b in bands:
        ax.axvspan(a, b, color=FAN_C, lw=0, zorder=0)
    for s in sp:
        ax.axvline(s, color=MUT, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.plot(xi, y, color=ACC, lw=1.8, zorder=3, solid_capstyle="round")
    ax.set_ylabel(sym, fontsize=11)
    ax.set_title(title, fontsize=9, color=MUT, loc="left", pad=26 if k < 3 else 6)
    ax.grid(axis="y", color="#ededed", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad = 0.12 * (hi - lo + 1e-12)
    ax.set_ylim(lo - pad, hi + pad)

# name the waves once, staggered so the close trio stays legible
top = axes[0, 0]
lo, hi = top.get_ylim()
for n, (s, nm) in enumerate(zip(sp, names)):
    top.annotate(nm, xy=(s, hi), xytext=(0, 3 + 10 * (n % 2)),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=8, color=MUT, annotation_clip=False)
for ax in axes[1]:
    ax.set_xlabel(r"$\xi = x/t$")

RAR = ", ".join(n for n in ("LF", "LS", "RS", "RF") if not kind[n])
SHK = ", ".join(n for n in ("LF", "LS", "RS", "RF") if kind[n])
fig.suptitle("Exact five-wave planar RMHD Riemann solution "
             "— a real rotor interface the seven-wave solver could not resolve",
             fontsize=11.5, x=0.010, ha="left", y=0.996)
sub = (r"$B^n = %.3f$      left $(\rho,\,p_{\rm tot},\,v^x,\,v^t,\,B^t) = "
       r"(%.3f,\,%.3f,\,%.3f,\,%.3f,\,%.3f)$      right $= "
       r"(%.3f,\,%.3f,\,%.3f,\,%.3f,\,%.3f)$" "\n"
       r"%s are rarefactions (shaded fans), %s is a shock;  CD is the contact."
       "\n"
       r"verified against the complete seven-wave jump conditions at "
       r"$\|f\|_\infty = %.2e$"
       % (Bn, L[0], L[1], L[2], L[3], L[5], R[0], R[1], R[2], R[3], R[5],
          RAR, SHK, float(z["full"])))
fig.text(0.010, 0.955, sub, fontsize=8.2, color=MUT, ha="left", va="top")
fig.tight_layout(rect=[0, 0, 1, 0.885])
out = "/Users/miler/Codes/HLLD/figs/planar_riemann_profile.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=170)
print("wrote", out)
print("fan bands:", ["%.3f..%.3f" % b for b in bands])
print("kinds: LF=%s LS=%s RS=%s RF=%s"
      % tuple("shock" if x else "rarefaction" for x in (kLF, kLS, kRS, kRF)))
