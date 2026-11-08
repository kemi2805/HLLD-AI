#!/usr/bin/env python
"""Where in the four-quadrant problem the two loss types appear (2026-11-04;
user: "where in 4Q do these appear?").  Every lost face of the 128^2 run (the
face table keeps them all), typed from its two states as in the anatomy study:
  A        slow-Alfven gap < 1e-3 and |Bn|/|B| <= 0.9 (the 0.99c shear lines)
  B weak   |Bn|/|B| > 0.9 and beta >= 1 (intermediate shocks)
  B strong |Bn|/|B| > 0.9 and beta < 1 (B_t cannot steer the slow wave)
drawn over log rho of the snapshot at each window's middle, plus all t > 0.1
in the self-similar frame (x/t, y/t); the gallery's examples are marked.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feats import load                                   # noqa: E402

RUN = os.path.join(os.path.dirname(HERE), "runs64", "ft3_128")
COL = {"A": "#D55E00", "B weak field": "#0072B2", "B strong field": "#009E73"}
LAB = {"A": "A: 0.99c shear, slow = Alfvén speed",
       "B weak field": "B, β ≥ 1: intermediate shocks",
       "B strong field": "B, β < 1: B_t cannot steer the slow wave"}
EX = {"A1": 62, "A2": 254, "A3": 458, "A4": 666, "B1": 1099, "B2": 215, "B3": 1069, "B4": 1348}
WIN = ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4))


def main():
    d = load(os.path.join(HERE, "data", "sample.npz"))
    m = (d["group"] == 4) & d["aligned"] & d["phys"]
    fn, beta, gap = d["f_fn_max"].astype(float), d["f_beta_min"].astype(float), d["gapAS"]
    kind = np.full(d["group"].size, "", dtype=object)
    kind[(gap < 1e-3) & (fn <= 0.9)] = "A"
    kind[(fn > 0.9) & (beta >= 1)] = "B weak field"
    kind[(fn > 0.9) & (beta < 1)] = "B strong field"
    x, y, t = d["x"], d["y"], d["t"]
    snaps = []
    for f in sorted(glob.glob(os.path.join(RUN, "snap_0*.npz"))):
        z = np.load(f, allow_pickle=True)
        snaps.append({k: z[k] for k in ("t", "x", "y", "rho")})
    # the examples' positions (their states matched to the face table)
    p = np.load(os.path.join(HERE, "data", "ledger", "population.npz"), allow_pickle=True)
    key = lambda U1, U2: np.ascontiguousarray(np.concatenate([U1, U2], axis=1)).view(np.dtype((np.void, 128))).ravel()
    where = {}
    for i, k in enumerate(key(d["U_L"], d["U_R"])):
        if d["group"][i] == 4:
            where.setdefault(k.tobytes(), []).append(i)
    exk = key(p["UL"], p["UR"])
    expos = {nm: where.get(exk[ln].tobytes(), []) for nm, ln in EX.items()}

    fig, ax = plt.subplots(1, 5, figsize=(24, 5.6), constrained_layout=True)
    for j, (t0, t1) in enumerate(WIN):
        a = ax[j]
        s = min(snaps, key=lambda s_: abs(float(s_["t"]) - 0.5 * (t0 + t1)))
        a.imshow(np.log10(s["rho"]).T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap="Greys",
                 interpolation="nearest", alpha=0.75)
        counts = []
        for k in COL:
            mm = m & (kind == k) & (t >= t0) & (t < t1)
            a.scatter(x[mm], y[mm], s=5, color=COL[k], lw=0, alpha=0.55)
            counts.append("%s %s" % ({"A": "A", "B weak field": "B weak", "B strong field": "B strong"}[k],
                                     format(int(mm.sum()), ",")))
        for nm, idx in expos.items():
            idx = [i for i in idx if t0 <= t[i] < t1]
            if idx:
                i = idx[0]
                off = {"A1": (10, 10), "A2": (12, -14), "A3": (10, 10), "A4": (10, 10),
                       "B1": (-34, 22), "B2": (-30, 14), "B3": (16, -22), "B4": (22, 8)}[nm]
                a.annotate(nm, (x[i], y[i]), xytext=off, textcoords="offset points", fontsize=9,
                           color="#000000", fontweight="bold",
                           arrowprops=dict(arrowstyle="-", color="#000000", lw=0.8))
        a.set_title("t %.1f–%.1f\nlost: %s" % (t0, t1, " · ".join(counts)), fontsize=9.5, color="#222222")
        a.set_xticks([-0.5, 0, 0.5]); a.set_yticks([-0.5, 0, 0.5])
        a.tick_params(labelsize=8, colors="#666666")
    a = ax[4]
    s = snaps[-1]
    a.imshow(np.log10(s["rho"]).T, origin="lower", extent=[-0.5 / 0.4, 0.5 / 0.4, -0.5 / 0.4, 0.5 / 0.4],
             cmap="Greys", interpolation="nearest", alpha=0.75)
    for k in COL:
        mm = m & (kind == k) & (t > 0.1)
        a.scatter(x[mm] / t[mm], y[mm] / t[mm], s=4, color=COL[k], lw=0, alpha=0.5, label=LAB[k])
    a.set_xlim(-1.25, 1.25); a.set_ylim(-1.25, 1.25)
    a.set_title("all t > 0.1, at (x/t, y/t) over ρ at t = 0.4", fontsize=9.5, color="#222222")
    a.set_xlabel("x / t", fontsize=8.5, color="#666666"); a.set_ylabel("y / t", fontsize=8.5, color="#666666")
    a.tick_params(labelsize=8, colors="#666666")
    a.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=8.5, loc="upper left", markerscale=3)
    fig.suptitle("Where the lost faces of the four-quadrant 128² run lie, by loss type "
                 "(grey: log10 ρ; labels: the gallery's examples; every face has a mirror twin across the diagonal)",
                 fontsize=11.5, color="#222222")
    out = os.path.join(HERE, "figs", "loss_where.png")
    fig.savefig(out, dpi=130)
    tot = {k: int((m & (kind == k)).sum()) for k in COL}
    print("lost faces by type:", tot, "untyped", int((m & (kind == "")).sum()))
    # how much of each type lies on the axis lines (|y| < 0.02 with x < 0, or |x| < 0.02 with y < 0)
    axis = ((np.abs(y) < 0.02) & (x < -0.05)) | ((np.abs(x) < 0.02) & (y < -0.05))
    centre = (np.abs(x) < 0.12) & (np.abs(y) < 0.12)
    head = (x > 0.15) & (y > 0.15)
    for k in COL:
        mm = m & (kind == k)
        print("  %-15s on the shear lines %.2f, centre box %.2f, jet head (x,y > 0.15) %.2f" % (
            k, (mm & axis).sum() / mm.sum(), (mm & centre & ~axis).sum() / mm.sum(), (mm & head).sum() / mm.sum()))
    print("wrote", out)


if __name__ == "__main__":
    main()
